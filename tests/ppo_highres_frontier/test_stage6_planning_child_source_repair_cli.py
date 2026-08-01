from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/create_ppo_stage6_planning_child_source_repair.py"
CONTINUATION_SCRIPT = (
    ROOT
    / "scripts/create_ppo_stage6_planning_child_source_repair_continuation.py"
)


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stage6_planning_child_source_repair_cli",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_continuation_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stage6_planning_child_source_repair_continuation_cli",
        CONTINUATION_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact() -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema_version": "stage6_planning_child_source_repair/v1",
        "mode": "same_run_exact_resume_after_reviewed_source_repair/v1",
        "formal_run_id": "s6-standard-single-r1-20260724T000124Z",
        "seed": 20260716,
    }
    artifact["canonical_sha256"] = __import__("hashlib").sha256(
        ArtifactStore.canonical_json_bytes(artifact)
    ).hexdigest()
    return artifact


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--stage-root",
        str(tmp_path / "s6"),
        "--run-id",
        "s6-standard-single-r1-20260724T000124Z",
        "--review-authorization",
        str(tmp_path / "authorization.json"),
        "--planning-warm-start",
        str(tmp_path / "planning-warm-start.json"),
        "--spec-review",
        str(tmp_path / "spec-review.json"),
        "--quality-review",
        str(tmp_path / "quality-review.json"),
        "--implementation-report",
        str(tmp_path / "implementer-report.md"),
        "--exact-replay-evidence",
        str(tmp_path / "exact-replay.json"),
        "--output",
        str(tmp_path / "s6/planning-child-source-repair.json"),
    ]


def _continuation_argv(
    tmp_path: Path,
    *,
    publish: bool = False,
) -> list[str]:
    stage_root = tmp_path / "run" / "s6"
    output = stage_root / "planning-child-source-repair-continuation.json"
    argv = [
        "--stage-root",
        str(stage_root),
        "--run-id",
        stage_root.parent.name,
        "--parent",
        str(stage_root / "planning-child-source-repair.json"),
        "--review-authorization",
        str(tmp_path / "authorization.json"),
        "--implementation-report",
        str(tmp_path / "implementer-report.md"),
        "--spec-review",
        str(tmp_path / "spec-review.md"),
        "--quality-review",
        str(tmp_path / "quality-review.md"),
        "--stage5-gate",
        str(tmp_path / "gate.json"),
        "--output",
        str(output),
    ]
    if publish:
        argv.append("--publish")
    return argv


def test_child_repair_preview_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    output = tmp_path / "s6/planning-child-source-repair.json"
    monkeypatch.setattr(
        cli,
        "create_planning_child_source_repair_preview",
        lambda **kwargs: _artifact(),
    )

    assert cli.main(_argv(tmp_path)) == 0

    assert not output.exists()
    rendered = json.loads(capsys.readouterr().out)
    assert (
        rendered["schema_version"]
        == "stage6_planning_child_source_repair/v1"
    )


def test_child_repair_cli_publishes_only_with_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    artifact = _artifact()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "create_planning_child_source_repair_preview",
        lambda **kwargs: artifact,
    )
    monkeypatch.setattr(
        cli,
        "publish_planning_child_source_repair_artifact",
        lambda value, **kwargs: (
            calls.append({"artifact": value, **kwargs})
            or Path(kwargs["output_path"])
        ),
    )

    assert cli.main([*_argv(tmp_path), "--publish"]) == 0

    assert len(calls) == 1
    assert calls[0]["artifact"] == artifact
    assert calls[0]["stage_root"] == tmp_path / "s6"
    assert calls[0]["output_path"] == (
        tmp_path / "s6/planning-child-source-repair.json"
    )


def test_continuation_preview_uses_recovery_anchor_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cli = _load_continuation_cli()
    stage_root = tmp_path / "run" / "s6"
    stage_root.mkdir(parents=True)
    parent = stage_root / "planning-child-source-repair.json"
    authorization = tmp_path / "authorization.json"
    anchor = object()
    handle = SimpleNamespace(
        canonical_record=lambda: {"authorized": True},
        require_current=lambda _label: None,
    )
    authority = SimpleNamespace(require_current=lambda _label: None)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "secure_read_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(payload=b"config"),
    )
    monkeypatch.setattr(
        cli,
        "verify_stage6_review_launch_authorization",
        lambda **_kwargs: handle,
    )
    monkeypatch.setattr(
        cli,
        "stage6_execution_identity",
        lambda **_kwargs: {"identity": True},
    )
    monkeypatch.setattr(
        cli,
        "validate_stage6_verified_review_authorization",
        lambda *_args, **_kwargs: {"authorized": True},
    )
    monkeypatch.setattr(
        cli,
        "verify_frozen_stage5_authority",
        lambda **_kwargs: authority,
    )
    monkeypatch.setattr(
        cli,
        "_stage6_current_immutable_bindings",
        lambda **_kwargs: {"immutable": True},
    )
    monkeypatch.setattr(
        cli,
        "inspect_planning_child_recovery_anchor",
        lambda **_kwargs: anchor,
        raising=False,
    )

    def build(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        assert kwargs["anchor"] is anchor
        return {"schema_version": "continuation-preview"}

    monkeypatch.setattr(
        cli,
        "build_planning_child_source_repair_continuation_artifact",
        build,
    )
    publish_capability = MappingProxyType(
        {"schema_version": "continuation-preview"}
    )

    def issue(
        artifact: dict[str, object],
        **kwargs: object,
    ) -> object:
        assert artifact == {"schema_version": "continuation-preview"}
        assert kwargs["anchor"] is anchor
        assert isinstance(kwargs["run_lease"], RunLease)
        return publish_capability

    monkeypatch.setattr(
        cli,
        "issue_planning_child_source_repair_continuation_publish_capability",
        issue,
    )

    with RunLease(stage_root.parent / ".stage6.lease") as run_lease:
        result = cli.create_planning_child_source_repair_continuation_preview(
            stage_root=stage_root,
            run_id=stage_root.parent.name,
            parent_path=parent,
            review_authorization_path=authorization,
            implementation_report_path=tmp_path / "report.md",
            spec_review_path=tmp_path / "spec.md",
            quality_review_path=tmp_path / "quality.md",
            run_lease=run_lease,
            stage5_gate_path=tmp_path / "gate.json",
        )

    assert result is publish_capability
    assert captured["parent_path"] == parent
    assert not (
        stage_root / "planning-child-source-repair-continuation.json"
    ).exists()


@pytest.mark.parametrize("publish", (False, True))
def test_continuation_cli_rejects_competing_run_lease_before_preview_or_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publish: bool,
) -> None:
    cli = _load_continuation_cli()
    stage_root = tmp_path / "run" / "s6"
    stage_root.mkdir(parents=True)
    output = stage_root / "planning-child-source-repair-continuation.json"
    calls: list[str] = []

    monkeypatch.setattr(
        cli,
        "create_planning_child_source_repair_continuation_preview",
        lambda **_kwargs: calls.append("preview") or _artifact(),
    )

    def publish_artifact(
        artifact: dict[str, object],
        **_kwargs: object,
    ) -> Path:
        del artifact
        calls.append("publish")
        output.write_bytes(b"unexpected")
        return output

    monkeypatch.setattr(
        cli,
        "publish_planning_child_source_repair_continuation_artifact",
        publish_artifact,
    )

    with RunLease(stage_root.parent / ".stage6.lease"):
        with pytest.raises(RunLeaseError, match="held|unavailable"):
            cli.main(_continuation_argv(tmp_path, publish=publish))

    assert calls == []
    assert not output.exists()
