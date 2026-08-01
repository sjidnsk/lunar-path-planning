from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


STAGE_ID = "ppo_highres_frontier_stage6_standard_training_eval/v1"
SINGLE_SEED_SCOPE = {
    "scope_kind": "single_seed_system_closure/v1",
    "seed": 20260716,
    "updates": 100,
}
DIFF_FORMAT = "git_diff_binary_full_index_tree_to_tree/v1"
REVIEW_SCHEMA = "stage6_fresh_review/v1"
FORMAL_RUN_ID = "s6-standard-single-r1-20260717T010203Z"
OTHER_FORMAL_RUN_ID = "s6-standard-single-r1-20260717T010204Z"


def _module() -> ModuleType:
    return importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6_review_authorization"
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(
    repo: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        input=input_bytes,
        capture_output=True,
        check=True,
        shell=False,
        env=environment,
    )


def _git_text(
    repo: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> str:
    return _git(repo, *arguments, input_bytes=input_bytes).stdout.decode(
        "ascii"
    ).strip()


def _tree_from_worktree(repo: Path) -> str:
    rows: list[str] = []
    for relative in ("config.json", "payload.txt"):
        blob = _git_text(repo, "hash-object", "-w", "--", relative)
        rows.append(f"100644 blob {blob}\t{relative}\n")
    return _git_text(
        repo,
        "mktree",
        input_bytes="".join(rows).encode("utf-8"),
    )


def _canonical_diff(repo: Path, base_commit: str, tree: str) -> bytes:
    return _git(
        repo,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-color",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        base_commit,
        tree,
        "--",
    ).stdout


def _execution_identity(
    *,
    base_commit: str,
    prospective_tree: str,
    config_bytes: bytes,
    source_bytes: bytes,
) -> dict[str, object]:
    changed_paths = ["payload.txt"]
    source_sha256 = hashlib.sha256(
        b"payload.txt\0" + source_bytes + b"\0"
    ).hexdigest()
    source_identity = {
        "schema_version": "stage6_current_source_set/v1",
        "source_set_sha256": source_sha256,
        "paths": [
            {
                "path": "payload.txt",
                "sha256": _sha256(source_bytes),
                "size_bytes": len(source_bytes),
            }
        ],
    }
    data_identity = {
        "schema_version": "stage6_data_catalog_identity/v1",
        "files": [],
        "catalog_sha256": "3" * 64,
    }
    environment_identity = {
        "schema_version": "stage6_environment_identity/test-v1",
        "machine": "fixture",
    }
    return {
        "schema_version": "stage6_execution_identity/v1",
        "base_commit": base_commit,
        "head_commit": base_commit,
        "prospective_git_tree": prospective_tree,
        "prospective_tree_sha256": _sha256(prospective_tree.encode("ascii")),
        "changed_paths": changed_paths,
        "changed_path_set_sha256": _canonical_sha256(changed_paths),
        "real_index_empty": True,
        "config_sha256": _sha256(config_bytes),
        "source_set_sha256": source_sha256,
        "data_sha256": _canonical_sha256(data_identity),
        "catalog_sha256": "3" * 64,
        "environment_identity": environment_identity,
        "environment_sha256": _canonical_sha256(environment_identity),
        "source_identity": source_identity,
        "data_identity": data_identity,
    }


def _artifact_binding(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _review_document(
    *,
    review_kind: str,
    execution_identity_sha256: str,
    prospective_tree: str,
    minor: int,
) -> dict[str, object]:
    prefix = "S" if review_kind == "spec_compliance" else "Q"
    findings = [
        {
            "finding_id": f"{prefix}-M{index + 1}",
            "severity": "minor",
            "summary": f"minor finding {index + 1}",
        }
        for index in range(minor)
    ]
    return {
        "schema_version": REVIEW_SCHEMA,
        "review_kind": review_kind,
        "stage_id": STAGE_ID,
        "formal_run_id": FORMAL_RUN_ID,
        "reviewed_execution_identity_sha256": execution_identity_sha256,
        "reviewed_prospective_git_tree": prospective_tree,
        "verdict": "PASS",
        "finding_counts": {
            "critical": 0,
            "important": 0,
            "minor": minor,
        },
        "findings": findings,
    }


def _review_binding(
    path: str,
    payload: bytes,
    document: dict[str, object],
) -> dict[str, object]:
    return {
        **_artifact_binding(path, payload),
        "review_kind": document["review_kind"],
        "stage_id": document["stage_id"],
        "formal_run_id": document["formal_run_id"],
        "reviewed_execution_identity_sha256": document[
            "reviewed_execution_identity_sha256"
        ],
        "reviewed_prospective_git_tree": document[
            "reviewed_prospective_git_tree"
        ],
        "verdict": document["verdict"],
        "finding_counts": copy.deepcopy(document["finding_counts"]),
    }


@dataclass
class _ReviewPackage:
    module: ModuleType
    repo: Path
    config_path: Path
    root: Path
    authorization_path: Path
    execution_identity_path: Path
    frozen_diff_path: Path
    spec_review_path: Path
    quality_review_path: Path
    spec_review: dict[str, object]
    quality_review: dict[str, object]
    authorization: dict[str, object]
    reviewed_identity: dict[str, object]
    identity_state: dict[str, object]
    identity_calls: list[tuple[Path, Path]]

    def rewrite_authorization(self) -> bytes:
        payload = _canonical_json_bytes(self.authorization)
        self.authorization_path.write_bytes(payload)
        return payload

    def rebind_execution_identity_file(self, payload: bytes) -> None:
        self.execution_identity_path.write_bytes(payload)
        digest = _sha256(payload)
        self.authorization["reviewed_execution_identity_sha256"] = digest
        self.authorization["execution_identity"]["sha256"] = digest
        self.authorization["execution_identity"]["size_bytes"] = len(payload)
        for review_name in ("spec_review", "quality_review"):
            document = copy.deepcopy(getattr(self, review_name))
            document["reviewed_execution_identity_sha256"] = digest
            setattr(self, review_name, document)
            self.rebind_review_file(
                review_name,
                document,
                bind_semantics=True,
            )

    def rebind_review_file(
        self,
        review_name: str,
        document: dict[str, object],
        *,
        bind_semantics: bool,
        canonical: bool = True,
    ) -> bytes:
        if review_name not in {"spec_review", "quality_review"}:
            raise AssertionError(f"unknown review fixture: {review_name}")
        path = getattr(self, f"{review_name}_path")
        payload = (
            _canonical_json_bytes(document)
            if canonical
            else json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        path.write_bytes(payload)
        binding = self.authorization[review_name]
        binding["sha256"] = _sha256(payload)
        binding["size_bytes"] = len(payload)
        if bind_semantics:
            for field in (
                "review_kind",
                "stage_id",
                "formal_run_id",
                "reviewed_execution_identity_sha256",
                "reviewed_prospective_git_tree",
                "verdict",
            ):
                binding[field] = document[field]
            binding["finding_counts"] = copy.deepcopy(document["finding_counts"])
        self.rewrite_authorization()
        return payload


@pytest.fixture
def review_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _ReviewPackage:
    module = _module()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "config", "user.name", "Stage6 Review Test")
    _git(repo, "config", "user.email", "stage6-review@example.invalid")

    config_path = repo / "config.json"
    config_bytes = b'{"stage":"stage6"}\n'
    config_path.write_bytes(config_bytes)
    payload_path = repo / "payload.txt"
    payload_path.write_bytes(b"base\n")
    _git(repo, "add", "--", "config.json", "payload.txt")
    _git(repo, "commit", "--quiet", "-m", "stage5 base")
    base_commit = _git_text(repo, "rev-parse", "HEAD")

    reviewed_source = b"reviewed\n"
    payload_path.write_bytes(reviewed_source)
    prospective_tree = _tree_from_worktree(repo)
    reviewed_identity = _execution_identity(
        base_commit=base_commit,
        prospective_tree=prospective_tree,
        config_bytes=config_bytes,
        source_bytes=reviewed_source,
    )
    identity_bytes = _canonical_json_bytes(reviewed_identity)
    frozen_diff = _canonical_diff(repo, base_commit, prospective_tree)
    identity_sha256 = _sha256(identity_bytes)
    spec_review = _review_document(
        review_kind="spec_compliance",
        execution_identity_sha256=identity_sha256,
        prospective_tree=prospective_tree,
        minor=2,
    )
    quality_review = _review_document(
        review_kind="quality_security",
        execution_identity_sha256=identity_sha256,
        prospective_tree=prospective_tree,
        minor=1,
    )
    spec_review_bytes = _canonical_json_bytes(spec_review)
    quality_review_bytes = _canonical_json_bytes(quality_review)

    root = tmp_path / "review-package"
    root.mkdir()
    execution_identity_path = root / "execution-identity.json"
    frozen_diff_path = root / "prospective.diff"
    spec_review_path = root / "spec-review.json"
    quality_review_path = root / "quality-review.json"
    execution_identity_path.write_bytes(identity_bytes)
    frozen_diff_path.write_bytes(frozen_diff)
    spec_review_path.write_bytes(spec_review_bytes)
    quality_review_path.write_bytes(quality_review_bytes)

    authorization = {
        "schema_version": "stage6_review_launch_authorization/v1",
        "stage_id": STAGE_ID,
        "formal_run_id": FORMAL_RUN_ID,
        "single_seed_scope": copy.deepcopy(SINGLE_SEED_SCOPE),
        "authorized": True,
        "base_commit": base_commit,
        "head_commit": base_commit,
        "reviewed_execution_identity": copy.deepcopy(reviewed_identity),
        "reviewed_execution_identity_sha256": identity_sha256,
        "execution_identity": _artifact_binding(
            execution_identity_path.name,
            identity_bytes,
        ),
        "reviewed_prospective_git_tree": prospective_tree,
        "prospective_tree_sha256": _sha256(prospective_tree.encode("ascii")),
        "changed_path_set_sha256": reviewed_identity[
            "changed_path_set_sha256"
        ],
        "source_set_sha256": reviewed_identity["source_set_sha256"],
        "config_sha256": reviewed_identity["config_sha256"],
        "data_sha256": reviewed_identity["data_sha256"],
        "environment_sha256": reviewed_identity["environment_sha256"],
        "frozen_diff": {
            **_artifact_binding(frozen_diff_path.name, frozen_diff),
            "format": DIFF_FORMAT,
        },
        "spec_review": _review_binding(
            spec_review_path.name,
            spec_review_bytes,
            spec_review,
        ),
        "quality_review": _review_binding(
            quality_review_path.name,
            quality_review_bytes,
            quality_review,
        ),
    }
    authorization_path = root / "launch-authorization.json"
    authorization_path.write_bytes(_canonical_json_bytes(authorization))

    identity_state: dict[str, object] = {
        "value": copy.deepcopy(reviewed_identity)
    }
    identity_calls: list[tuple[Path, Path]] = []

    def current_identity(
        *,
        repo_root: str | Path,
        config_path: str | Path,
        effective_config_bytes: bytes | None = None,
    ):
        identity_calls.append((Path(repo_root), Path(config_path)))
        identity_state["effective_config_bytes"] = effective_config_bytes
        return copy.deepcopy(identity_state["value"])

    monkeypatch.setattr(module, "STAGE5_COMMIT", base_commit)
    monkeypatch.setattr(module, "stage6_execution_identity", current_identity)
    monkeypatch.setattr(
        module,
        "_review_package_drive_is_allowed",
        lambda _path: True,
    )
    return _ReviewPackage(
        module=module,
        repo=repo,
        config_path=config_path,
        root=root,
        authorization_path=authorization_path,
        execution_identity_path=execution_identity_path,
        frozen_diff_path=frozen_diff_path,
        spec_review_path=spec_review_path,
        quality_review_path=quality_review_path,
        spec_review=spec_review,
        quality_review=quality_review,
        authorization=authorization,
        reviewed_identity=reviewed_identity,
        identity_state=identity_state,
        identity_calls=identity_calls,
    )


def _verify(package: _ReviewPackage) -> object:
    return package.module.verify_stage6_review_launch_authorization(
        authorization_path=package.authorization_path,
        repo_root=package.repo,
        config_path=package.config_path,
        expected_run_id=FORMAL_RUN_ID,
    )


def test_review_authorization_binds_and_revalidates_effective_config_bytes(
    review_package: _ReviewPackage,
) -> None:
    effective = b'{"effective":"planning"}\n'

    handle = (
        review_package.module.verify_stage6_review_launch_authorization(
            authorization_path=review_package.authorization_path,
            repo_root=review_package.repo,
            config_path=review_package.config_path,
            expected_run_id=FORMAL_RUN_ID,
            effective_config_bytes=effective,
        )
    )
    assert review_package.identity_state["effective_config_bytes"] == effective

    review_package.identity_state["effective_config_bytes"] = None
    handle.require_current("effective config revalidation")
    assert review_package.identity_state["effective_config_bytes"] == effective


def test_verified_authorization_binds_full_review_identity_and_frozen_evidence(
    review_package: _ReviewPackage,
) -> None:
    before = {
        path.name: path.read_bytes()
        for path in review_package.root.iterdir()
        if path.is_file()
    }
    authorization_bytes = review_package.authorization_path.read_bytes()

    handle = _verify(review_package)
    assert isinstance(handle, review_package.module.Stage6ReviewAuthorizationHandle)
    verified = handle.canonical_record()

    identity = review_package.reviewed_identity
    assert verified == {
        "schema_version": "stage6_verified_review_launch_authorization/v1",
        "stage_id": STAGE_ID,
        "formal_run_id": FORMAL_RUN_ID,
        "single_seed_scope": SINGLE_SEED_SCOPE,
        "authorized": True,
        "base_commit": identity["base_commit"],
        "authorization_file_sha256": _sha256(authorization_bytes),
        "authorization_file_size_bytes": len(authorization_bytes),
        "review_identity_sha256": review_package.authorization[
            "reviewed_execution_identity_sha256"
        ],
        "reviewed_prospective_git_tree": identity["prospective_git_tree"],
        "prospective_tree_sha256": identity["prospective_tree_sha256"],
        "changed_path_set_sha256": identity["changed_path_set_sha256"],
        "source_set_sha256": identity["source_set_sha256"],
        "config_sha256": identity["config_sha256"],
        "data_sha256": identity["data_sha256"],
        "environment_sha256": identity["environment_sha256"],
        "frozen_diff_sha256": review_package.authorization["frozen_diff"][
            "sha256"
        ],
        "frozen_diff_size_bytes": review_package.authorization["frozen_diff"][
            "size_bytes"
        ],
        "spec_review_sha256": review_package.authorization["spec_review"][
            "sha256"
        ],
        "quality_review_sha256": review_package.authorization["quality_review"][
            "sha256"
        ],
    }
    assert dict(handle.record) == verified
    with pytest.raises(TypeError):
        handle.record["authorized"] = False
    detached = handle.canonical_record()
    detached["authorized"] = False
    assert handle.canonical_record()["authorized"] is True
    with pytest.raises(TypeError):
        json.dumps(handle)
    handle.require_current("test review package remains current")
    assert review_package.identity_calls == [
        (review_package.repo, review_package.config_path),
        (review_package.repo, review_package.config_path),
    ]
    assert {
        path.name: path.read_bytes()
        for path in review_package.root.iterdir()
        if path.is_file()
    } == before
    _git(review_package.repo, "diff", "--cached", "--quiet", "--exit-code")


def test_review_authorization_handle_exposes_exact_read_only_evidence_paths(
    review_package: _ReviewPackage,
) -> None:
    handle = _verify(review_package)

    assert handle.evidence_paths == (
        review_package.authorization_path.resolve(),
        review_package.execution_identity_path.resolve(),
        review_package.frozen_diff_path.resolve(),
        review_package.spec_review_path.resolve(),
        review_package.quality_review_path.resolve(),
    )
    assert isinstance(handle.evidence_paths, tuple)
    with pytest.raises((AttributeError, TypeError)):
        handle.evidence_paths = ()

    records = handle.evidence_records
    assert tuple(record.path for record in records) == handle.evidence_paths
    assert tuple(record.sha256 for record in records) == tuple(
        _sha256(path.read_bytes()) for path in handle.evidence_paths
    )
    assert tuple(record.size_bytes for record in records) == tuple(
        len(path.read_bytes()) for path in handle.evidence_paths
    )
    assert isinstance(records, tuple)
    with pytest.raises((AttributeError, TypeError)):
        records[0].sha256 = "f" * 64


def test_authorization_is_bound_to_exact_expected_formal_run_id(
    review_package: _ReviewPackage,
) -> None:
    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="formal run id",
    ):
        review_package.module.verify_stage6_review_launch_authorization(
            authorization_path=review_package.authorization_path,
            repo_root=review_package.repo,
            config_path=review_package.config_path,
            expected_run_id=OTHER_FORMAL_RUN_ID,
        )

    assert review_package.identity_calls == []


def test_same_changed_path_set_with_one_byte_source_drift_is_rejected_before_run_root(
    review_package: _ReviewPackage,
    tmp_path: Path,
) -> None:
    source_path = review_package.repo / "payload.txt"
    drifted_source = b"reviewed!\n"
    source_path.write_bytes(drifted_source)
    drifted_tree = _tree_from_worktree(review_package.repo)
    review_package.identity_state["value"] = _execution_identity(
        base_commit=str(review_package.reviewed_identity["base_commit"]),
        prospective_tree=drifted_tree,
        config_bytes=review_package.config_path.read_bytes(),
        source_bytes=drifted_source,
    )
    run_root = tmp_path / "must-not-exist"

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="current execution identity differs from reviewed identity",
    ):
        _verify(review_package)

    assert review_package.identity_state["value"]["changed_paths"] == (
        review_package.reviewed_identity["changed_paths"]
    )
    assert drifted_tree != review_package.reviewed_identity["prospective_git_tree"]
    assert not run_root.exists()
    _git(review_package.repo, "diff", "--cached", "--quiet", "--exit-code")


def test_replaced_execution_identity_file_is_rejected(
    review_package: _ReviewPackage,
) -> None:
    replacement = copy.deepcopy(review_package.reviewed_identity)
    replacement["config_sha256"] = "f" * 64
    review_package.execution_identity_path.write_bytes(
        json.dumps(replacement, sort_keys=True).encode("utf-16")
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="execution identity SHA-256 drift",
    ):
        _verify(review_package)


@pytest.mark.parametrize("encoding", ("utf-16", "noncanonical-utf-8"))
def test_execution_identity_file_must_be_utf8_canonical_json(
    review_package: _ReviewPackage,
    encoding: str,
) -> None:
    if encoding == "utf-16":
        payload = json.dumps(
            review_package.reviewed_identity,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-16")
    else:
        payload = json.dumps(
            review_package.reviewed_identity,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    review_package.rebind_execution_identity_file(payload)

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="execution identity is not canonical JSON",
    ):
        _verify(review_package)


def test_reviewed_execution_identity_field_set_is_exact(
    review_package: _ReviewPackage,
) -> None:
    identity = copy.deepcopy(review_package.reviewed_identity)
    identity["unexpected"] = "not authorized"
    review_package.authorization["reviewed_execution_identity"] = copy.deepcopy(
        identity
    )
    review_package.identity_state["value"] = copy.deepcopy(identity)
    review_package.rebind_execution_identity_file(_canonical_json_bytes(identity))

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="reviewed execution identity field set drift",
    ):
        _verify(review_package)

    assert review_package.identity_calls == []


def test_frozen_diff_byte_drift_is_rejected(review_package: _ReviewPackage) -> None:
    review_package.frozen_diff_path.write_bytes(
        review_package.frozen_diff_path.read_bytes() + b"# drift\n"
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="frozen diff SHA-256 drift",
    ):
        _verify(review_package)


def test_review_with_important_finding_is_rejected(
    review_package: _ReviewPackage,
) -> None:
    review_package.authorization["quality_review"]["finding_counts"][
        "important"
    ] = 1
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match=(
            "review must be PASS with zero Critical and Important findings"
            "|quality review finding counts content/binding drift"
        ),
    ):
        _verify(review_package)


def test_review_content_fail_cannot_be_hidden_by_pass_binding(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["verdict"] = "FAIL"
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=False,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review.*verdict.*binding|quality review.*content.*binding",
    ):
        _verify(review_package)


def test_review_content_important_count_cannot_be_hidden_by_zero_binding(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["finding_counts"] = {
        "critical": 0,
        "important": 1,
        "minor": 1,
    }
    document["findings"].append(
        {
            "finding_id": "Q-I1",
            "severity": "important",
            "summary": "important finding hidden by metadata",
        }
    )
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=False,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review.*finding counts.*binding|quality review.*content.*binding",
    ):
        _verify(review_package)


def test_spec_and_quality_review_kinds_cannot_be_interchanged(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    review_package.rebind_review_file(
        "spec_review",
        document,
        bind_semantics=True,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="spec review.*kind",
    ):
        _verify(review_package)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("formal_run_id", OTHER_FORMAL_RUN_ID, "formal run id"),
        ("reviewed_prospective_git_tree", "f" * 40, "reviewed tree"),
        (
            "reviewed_execution_identity_sha256",
            "e" * 64,
            "execution identity",
        ),
        ("stage_id", "wrong-stage/v1", "stage id"),
    ),
)
def test_review_content_must_bind_authorized_run_tree_identity_and_stage(
    review_package: _ReviewPackage,
    field: str,
    value: str,
    message: str,
) -> None:
    document = copy.deepcopy(review_package.spec_review)
    document[field] = value
    review_package.rebind_review_file(
        "spec_review",
        document,
        bind_semantics=True,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match=message,
    ):
        _verify(review_package)


def test_review_file_must_be_canonical_utf8_json(
    review_package: _ReviewPackage,
) -> None:
    review_package.rebind_review_file(
        "spec_review",
        copy.deepcopy(review_package.spec_review),
        bind_semantics=False,
        canonical=False,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="spec review is not canonical JSON",
    ):
        _verify(review_package)


def test_review_document_rejects_extra_fields(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.spec_review)
    document["unexpected"] = "not reviewed"
    review_package.rebind_review_file(
        "spec_review",
        document,
        bind_semantics=False,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="spec review field set drift",
    ):
        _verify(review_package)


@pytest.mark.parametrize("minor_count", (0, 1))
def test_review_document_requires_findings_for_zero_and_nonzero_counts(
    review_package: _ReviewPackage,
    minor_count: int,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["finding_counts"] = {
        "critical": 0,
        "important": 0,
        "minor": minor_count,
    }
    document.pop("findings")
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=True,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review field set drift",
    ):
        _verify(review_package)


def test_review_document_accepts_explicit_empty_findings_for_zero_counts(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["finding_counts"] = {
        "critical": 0,
        "important": 0,
        "minor": 0,
    }
    document["findings"] = []
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=True,
    )

    handle = _verify(review_package)

    assert handle.canonical_record()["quality_review_sha256"] == (
        review_package.authorization["quality_review"]["sha256"]
    )


def test_review_document_rejects_duplicate_finding_ids(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["finding_counts"]["minor"] = 2
    document["findings"].append(copy.deepcopy(document["findings"][0]))
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=True,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review finding is invalid",
    ):
        _verify(review_package)


def test_review_finding_fields_and_counts_are_exact(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["findings"][0]["unexpected"] = "not allowed"
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=False,
    )
    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review finding.*field set drift",
    ):
        _verify(review_package)

    document = copy.deepcopy(review_package.quality_review)
    document["findings"][0]["severity"] = "important"
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=False,
    )
    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review finding counts do not match findings",
    ):
        _verify(review_package)


def test_review_content_fail_is_not_launch_authority_even_when_binding_matches(
    review_package: _ReviewPackage,
) -> None:
    document = copy.deepcopy(review_package.quality_review)
    document["verdict"] = "FAIL"
    review_package.rebind_review_file(
        "quality_review",
        document,
        bind_semantics=True,
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="PASS with zero Critical and Important",
    ):
        _verify(review_package)


def test_noncanonical_authorization_json_is_rejected(
    review_package: _ReviewPackage,
) -> None:
    review_package.authorization_path.write_bytes(
        json.dumps(review_package.authorization, sort_keys=True).encode("utf-8")
    )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="authorization is not canonical JSON",
    ):
        _verify(review_package)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_authorization_rejects_nonfinite_json_constants(
    review_package: _ReviewPackage,
    constant: str,
) -> None:
    payload = review_package.authorization_path.read_bytes().replace(
        b'"seed": 20260716',
        f'"seed": {constant}'.encode("ascii"),
        1,
    )
    review_package.authorization_path.write_bytes(payload)

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="authorization.*non-finite|authorization.*invalid JSON",
    ):
        _verify(review_package)


def test_authorization_rejects_duplicate_json_keys(
    review_package: _ReviewPackage,
) -> None:
    payload = review_package.authorization_path.read_bytes().replace(
        b'"authorized": true,',
        b'"authorized": true,\n  "authorized": true,',
        1,
    )
    review_package.authorization_path.write_bytes(payload)

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="authorization.*duplicate JSON key|authorization.*invalid JSON",
    ):
        _verify(review_package)


@pytest.mark.parametrize(
    ("field", "value"),
    (("seed", 20260716.0), ("updates", 100.0)),
)
def test_single_seed_scope_requires_exact_integer_types(
    review_package: _ReviewPackage,
    field: str,
    value: float,
) -> None:
    review_package.authorization["single_seed_scope"][field] = value
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="single-seed scope",
    ):
        _verify(review_package)


def test_execution_identity_and_review_reject_duplicate_keys_and_nonfinite_values(
    review_package: _ReviewPackage,
) -> None:
    identity_payload = review_package.execution_identity_path.read_bytes().replace(
        b'"real_index_empty": true,',
        b'"real_index_empty": true,\n  "real_index_empty": true,',
        1,
    )
    review_package.execution_identity_path.write_bytes(identity_payload)
    identity_digest = _sha256(identity_payload)
    binding = review_package.authorization["execution_identity"]
    binding["sha256"] = identity_digest
    binding["size_bytes"] = len(identity_payload)
    review_package.authorization[
        "reviewed_execution_identity_sha256"
    ] = identity_digest
    for review_name in ("spec_review", "quality_review"):
        document = copy.deepcopy(getattr(review_package, review_name))
        document["reviewed_execution_identity_sha256"] = identity_digest
        setattr(review_package, review_name, document)
        review_package.rebind_review_file(
            review_name,
            document,
            bind_semantics=True,
        )
    review_package.rewrite_authorization()
    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="execution identity.*duplicate JSON key|execution identity.*invalid JSON",
    ):
        _verify(review_package)

    review_package.rebind_execution_identity_file(
        _canonical_json_bytes(review_package.reviewed_identity)
    )
    review_payload = review_package.quality_review_path.read_bytes().replace(
        b'"minor": 1',
        b'"minor": NaN',
        1,
    )
    review_package.quality_review_path.write_bytes(review_payload)
    review_binding = review_package.authorization["quality_review"]
    review_binding["sha256"] = _sha256(review_payload)
    review_binding["size_bytes"] = len(review_payload)
    review_package.rewrite_authorization()
    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="quality review.*non-finite|quality review.*invalid JSON",
    ):
        _verify(review_package)


def test_local_canonical_json_dump_rejects_nonfinite_values(
    review_package: _ReviewPackage,
) -> None:
    with pytest.raises((ValueError, review_package.module.Stage6ReviewAuthorizationError)):
        review_package.module._canonical_json_bytes({"value": float("nan")})


@pytest.mark.parametrize("malicious_path", ({"path": "payload.txt"}, ["payload.txt"]))
def test_changed_paths_reject_non_string_members_without_leaking_type_error(
    review_package: _ReviewPackage,
    malicious_path: object,
) -> None:
    identity = copy.deepcopy(review_package.reviewed_identity)
    identity["changed_paths"] = [malicious_path]
    identity["changed_path_set_sha256"] = _canonical_sha256(
        identity["changed_paths"]
    )
    review_package.authorization["reviewed_execution_identity"] = copy.deepcopy(
        identity
    )
    review_package.identity_state["value"] = copy.deepcopy(identity)
    review_package.rebind_execution_identity_file(_canonical_json_bytes(identity))

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="reviewed changed path set",
    ):
        _verify(review_package)


@pytest.mark.parametrize(
    "reference",
    ("../outside.json", "C:/outside.json", "/outside.json"),
)
def test_execution_identity_reference_escape_is_rejected(
    review_package: _ReviewPackage,
    reference: str,
) -> None:
    review_package.authorization["execution_identity"]["path"] = reference
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="relative reference",
    ):
        _verify(review_package)


@pytest.mark.parametrize(
    "member_name",
    (
        "launch-authorization.json",
        "execution-identity.json",
        "prospective.diff",
        "spec-review.json",
        "quality-review.json",
    ),
)
def test_every_authority_member_rejects_hardlinks(
    review_package: _ReviewPackage,
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
) -> None:
    member = review_package.root / member_name
    alias = review_package.root / f"{member_name}.hardlink"
    try:
        os.link(member, alias)
    except OSError:
        original_secure_read = review_package.module._secure_read_bytes

        def reject_via_controlled_seam(path, **kwargs):
            if Path(path) == member:
                raise review_package.module.PathSecurityError(
                    "controlled hard-link seam"
                )
            return original_secure_read(path, **kwargs)

        monkeypatch.setattr(
            review_package.module,
            "_secure_read_bytes",
            reject_via_controlled_seam,
        )

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="secure single-link read failed",
    ):
        _verify(review_package)


def test_symlinked_execution_identity_reference_is_rejected(
    review_package: _ReviewPackage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = review_package.root / "execution-identity-link.json"
    try:
        os.symlink(review_package.execution_identity_path, alias)
    except OSError:
        alias.write_bytes(review_package.execution_identity_path.read_bytes())
        original_secure_read = review_package.module._secure_read_bytes

        def reject_via_controlled_seam(path, **kwargs):
            if Path(path) == alias:
                raise review_package.module.PathSecurityError(
                    "controlled reparse seam"
                )
            return original_secure_read(path, **kwargs)

        monkeypatch.setattr(
            review_package.module,
            "_secure_read_bytes",
            reject_via_controlled_seam,
        )
    review_package.authorization["execution_identity"]["path"] = alias.name
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="secure single-link read failed",
    ):
        _verify(review_package)


def test_reference_size_drift_is_rejected(review_package: _ReviewPackage) -> None:
    review_package.authorization["frozen_diff"]["size_bytes"] += 1
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="frozen diff size drift",
    ):
        _verify(review_package)


def test_handle_revalidation_securely_rereads_every_member_and_recomputes_identity_and_diff(
    review_package: _ReviewPackage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _verify(review_package)
    secure_reads: list[Path] = []
    diff_calls: list[tuple[Path, str, str]] = []
    original_read = review_package.module._secure_read_bytes
    original_diff = review_package.module._canonical_prospective_diff

    def tracked_read(path, **kwargs):
        secure_reads.append(Path(path))
        return original_read(path, **kwargs)

    def tracked_diff(repo, *, base_commit, reviewed_tree):
        diff_calls.append((Path(repo), base_commit, reviewed_tree))
        return original_diff(
            repo,
            base_commit=base_commit,
            reviewed_tree=reviewed_tree,
        )

    monkeypatch.setattr(review_package.module, "_secure_read_bytes", tracked_read)
    monkeypatch.setattr(
        review_package.module,
        "_canonical_prospective_diff",
        tracked_diff,
    )

    handle.require_current("all evidence revalidation")

    assert set(secure_reads) == {
        review_package.authorization_path,
        review_package.execution_identity_path,
        review_package.frozen_diff_path,
        review_package.spec_review_path,
        review_package.quality_review_path,
    }
    assert diff_calls == [
        (
            review_package.repo,
            str(review_package.reviewed_identity["base_commit"]),
            str(review_package.reviewed_identity["prospective_git_tree"]),
        )
    ]
    assert review_package.identity_calls == [
        (review_package.repo, review_package.config_path),
        (review_package.repo, review_package.config_path),
    ]


@pytest.mark.parametrize(
    "member_name",
    (
        "launch-authorization.json",
        "execution-identity.json",
        "prospective.diff",
        "spec-review.json",
        "quality-review.json",
    ),
)
def test_handle_revalidation_rejects_member_byte_drift(
    review_package: _ReviewPackage,
    member_name: str,
) -> None:
    handle = _verify(review_package)
    member = review_package.root / member_name
    member.write_bytes(member.read_bytes() + b"drift")

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="member byte drift.*changed|member byte drift.*revalidation failed",
    ):
        handle.require_current("member byte drift")


def test_handle_revalidation_rejects_identical_byte_file_identity_replacement(
    review_package: _ReviewPackage,
) -> None:
    handle = _verify(review_package)
    member = review_package.execution_identity_path
    replacement = review_package.root / "identity-replacement.tmp"
    replacement.write_bytes(member.read_bytes())
    os.replace(replacement, member)

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="identity replacement.*identity changed|identity replacement.*changed",
    ):
        handle.require_current("identity replacement")


def test_handle_revalidation_rejects_new_hardlink_alias(
    review_package: _ReviewPackage,
) -> None:
    handle = _verify(review_package)
    alias = review_package.root / "quality-review.hardlink"
    try:
        os.link(review_package.quality_review_path, alias)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="hardlink drift.*revalidation failed|hardlink drift.*single-link",
    ):
        handle.require_current("hardlink drift")


def test_handle_revalidation_rejects_current_execution_identity_and_tree_drift(
    review_package: _ReviewPackage,
) -> None:
    handle = _verify(review_package)
    drifted = copy.deepcopy(review_package.reviewed_identity)
    drifted["source_set_sha256"] = "f" * 64
    review_package.identity_state["value"] = drifted

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="execution identity drift.*current execution identity",
    ):
        handle.require_current("execution identity drift")


def test_authorization_field_set_is_exact(review_package: _ReviewPackage) -> None:
    review_package.authorization["unexpected"] = "not allowed"
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="authorization field set drift",
    ):
        _verify(review_package)


@pytest.mark.parametrize(
    "field",
    (
        "prospective_tree_sha256",
        "changed_path_set_sha256",
        "source_set_sha256",
        "config_sha256",
        "data_sha256",
        "environment_sha256",
    ),
)
def test_redundant_authorization_hash_must_match_reviewed_identity(
    review_package: _ReviewPackage,
    field: str,
) -> None:
    review_package.authorization[field] = "f" * 64
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="authorization identity binding drift",
    ):
        _verify(review_package)


@pytest.mark.parametrize("field", ("base_commit", "head_commit"))
def test_base_and_head_must_both_be_stage5_commit(
    review_package: _ReviewPackage,
    field: str,
) -> None:
    review_package.authorization[field] = "f" * 40
    review_package.rewrite_authorization()

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="Stage 5 commit binding drift",
    ):
        _verify(review_package)


def test_real_git_index_must_be_empty(review_package: _ReviewPackage) -> None:
    (review_package.repo / "payload.txt").write_bytes(b"staged drift\n")
    _git(review_package.repo, "add", "--", "payload.txt")

    with pytest.raises(
        review_package.module.Stage6ReviewAuthorizationError,
        match="real Git index must be empty",
    ):
        _verify(review_package)


def test_non_d_path_requires_explicit_private_drive_seam(tmp_path: Path) -> None:
    module = _module()
    authorization = Path("C:/stage6-review-policy-test/launch-authorization.json")

    with pytest.raises(
        module.Stage6ReviewAuthorizationError,
        match="D drive review package",
    ):
        module.verify_stage6_review_launch_authorization(
            authorization_path=authorization,
            repo_root=tmp_path,
            config_path=tmp_path / "config.json",
            expected_run_id=FORMAL_RUN_ID,
        )


def test_public_api_has_no_bypass_or_output_parameters_and_git_subprocesses_are_checked() -> None:
    module = _module()
    assert "Stage6ReviewAuthorizationHandle" in module.__all__
    signature = inspect.signature(module.verify_stage6_review_launch_authorization)
    assert tuple(signature.parameters) == (
        "authorization_path",
        "repo_root",
        "config_path",
        "expected_run_id",
        "effective_config_bytes",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not {
        "identity_provider",
        "diff_provider",
        "output_root",
        "force",
        "skip",
        "fake",
    } & set(signature.parameters)
    handle_signature = inspect.signature(
        module.Stage6ReviewAuthorizationHandle.require_current
    )
    assert tuple(handle_signature.parameters) == ("self", "label")

    tree = ast.parse(inspect.getsource(module))
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert run_calls
    for call in run_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords.get("check"), ast.Constant)
        assert keywords["check"].value is True
        assert "shell" not in keywords or (
            isinstance(keywords["shell"], ast.Constant)
            and keywords["shell"].value is False
        )
