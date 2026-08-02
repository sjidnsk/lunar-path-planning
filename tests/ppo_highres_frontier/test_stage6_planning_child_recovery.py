from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from lunar_exploration_ppo.workflows import (
    stage6_planning_child_recovery as recovery,
)
from test_stage6_planning_child_source_repair_continuation import (
    _continuation_preview,
    _publish_successor_chain,
)
from test_stage6_workflow import (
    ROOT,
    _append_task5_accepted_through_u90_restart_tail,
    _append_task5_accepted_u86_restart_tail,
    _append_task5_resource_pre,
    _append_task5_resource_segment,
    _append_task5_u86_crash_suffix,
    _finalize_task5_planning_child_terminal_fixture,
    _run_task5_planning_child_semantic_verifier,
    _task5_terminal_resource,
)


def _published_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retained_checkpoint_updates: tuple[int, ...] = (84,),
):
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        _authorization_path,
        artifact,
    ) = _continuation_preview(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=retained_checkpoint_updates,
    )
    continuation_path = _publish_successor_chain(
        fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        artifact=artifact,
    )
    return (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    )


def _issue(
    fixture,
    continuation_path: Path,
    *,
    identity: dict[str, object],
    review: dict[str, object],
    immutable: dict[str, object],
) -> recovery.PlanningChildRecoveryCapability:
    return recovery.issue_planning_child_recovery_capability(
        stage_root=fixture.stage,
        parent_artifact_path=fixture.child_path,
        continuation_artifact_path=continuation_path,
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
    )


def _cursor_tuple(
    capability: recovery.PlanningChildRecoveryCapability,
) -> tuple[int, int, int, str, int, int, int | None]:
    cursor = capability.resume_cursor
    return (
        cursor.last_accepted_update,
        cursor.next_update,
        cursor.next_attempt,
        cursor.next_transaction_key,
        cursor.next_resource_segment_index,
        cursor.latest_complete_checkpoint_update,
        cursor.pending_pre_attempt,
    )


def _prepare_u100_terminal_evidence(
    fixture,
    capability: recovery.PlanningChildRecoveryCapability,
    *,
    identity: dict[str, object],
    review: dict[str, object],
    immutable: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        build_planning_child_acceptance_profile,
    )

    plain_identity = recovery._plain_json(identity)
    plain_review = recovery._plain_json(review)
    plain_immutable = recovery._plain_json(immutable)
    assert isinstance(plain_identity, dict)
    assert isinstance(plain_review, dict)
    assert isinstance(plain_immutable, dict)
    fixture.current_identity = plain_identity
    fixture.current_review = plain_review
    fixture.current_immutable = plain_immutable
    fixture.recovery_capability = capability
    monkeypatch.setattr(
        fixture.module,
        "stage6_execution_identity",
        lambda **_kwargs: copy.deepcopy(plain_identity),
    )
    monkeypatch.setattr(
        fixture.module,
        "validate_stage6_verified_review_authorization",
        lambda value, **_kwargs: recovery._plain_json(value),
    )
    performance = fixture.module.decide_performance_advantage(
        ppo_ci95_low=0.80,
        gain_over_cost_ci95_high=0.90,
    )
    fixture.acceptance = (
        standard_training.build_stage6_acceptance_artifacts(
            global_best=fixture.global_best["record"],
            performance_advantage_established=performance.established,
            performance_claim=performance.claim,
            ppo_ci95_low=performance.ppo_ci95_low,
            gain_over_cost_ci95_high=(
                performance.gain_over_cost_ci95_high
            ),
            final_evaluation_count=10,
            final_episode_count=640,
            checkpoint_receipt_count=26,
            immutable_bindings=plain_immutable,
            source_repair_binding=None,
            planning_child_source_repair_binding=(
                capability.evidence_binding()
            ),
            acceptance_profile=(
                build_planning_child_acceptance_profile()
            ),
        )
    )
    _run_task5_planning_child_semantic_verifier(fixture)
    _finalize_task5_planning_child_terminal_fixture(fixture)


def _prepare_u100_preterminal_evidence(
    fixture,
    capability: recovery.PlanningChildRecoveryCapability,
    *,
    identity: dict[str, object],
    review: dict[str, object],
    immutable: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    authority_root: Path,
) -> None:
    from lunar_exploration_ppo.ppo import standard_training
    from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
        build_planning_child_acceptance_profile,
    )
    from test_stage6_workflow import (
        _active_execution_capability,
    )

    fixture.current_identity = recovery._plain_json(identity)
    fixture.current_review = recovery._plain_json(review)
    fixture.current_immutable = recovery._plain_json(immutable)
    fixture.recovery_capability = capability
    monkeypatch.setattr(
        fixture.module,
        "stage6_execution_identity",
        lambda **_kwargs: copy.deepcopy(fixture.current_identity),
    )
    monkeypatch.setattr(
        fixture.module,
        "validate_stage6_verified_review_authorization",
        lambda value, **_kwargs: recovery._plain_json(value),
    )
    performance = fixture.module.decide_performance_advantage(
        ppo_ci95_low=0.80,
        gain_over_cost_ci95_high=0.90,
    )
    fixture.acceptance = standard_training.build_stage6_acceptance_artifacts(
        global_best=fixture.global_best["record"],
        performance_advantage_established=performance.established,
        performance_claim=performance.claim,
        ppo_ci95_low=performance.ppo_ci95_low,
        gain_over_cost_ci95_high=performance.gain_over_cost_ci95_high,
        final_evaluation_count=10,
        final_episode_count=640,
        checkpoint_receipt_count=26,
        immutable_bindings=fixture.current_immutable,
        source_repair_binding=None,
        planning_child_source_repair_binding=capability.evidence_binding(),
        acceptance_profile=build_planning_child_acceptance_profile(),
    )
    semantic_result = _run_task5_planning_child_semantic_verifier(fixture)
    with fixture.terminal_recovery._capture_preterminal_evidence_handle(
        fixture.stage
    ) as evidence:
        semantic = fixture.terminal_recovery._bind_planning_child_semantic_result(
            {
                "schema_version": fixture.terminal_recovery.SEMANTIC_SCHEMA,
                **semantic_result,
                "evidence_binding": evidence.binding,
            },
            capability=capability,
        )
    terminal_artifacts = {
        "summary.json": fixture.module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["summary"]
        ),
        "routing.json": fixture.module.ArtifactStore.canonical_json_bytes(
            fixture.acceptance["routing"]
        ),
        **fixture.acceptance["reports"],
    }
    with _active_execution_capability(
        authority_root,
        monkeypatch,
        run_root=fixture.stage.parent,
        effective_config_bytes=fixture.effective_config_bytes,
        planning_warm_start_path=fixture.warm_start_path,
        planning_child_source_repair_path=fixture.child_path,
        formal_run_id=fixture.stage.parent.name,
    ) as active:
        fixture.terminal_recovery.write_stage6_preterminal_acceptance(
            stage_root=fixture.stage,
            semantic_result=semantic,
            immutable_bindings=fixture.current_immutable,
            global_checkpoint_identity=fixture.global_best,
            terminal_artifacts=terminal_artifacts,
            execution_capability=active["capability"],
        )
    detection = fixture.terminal_recovery.detect_stage6_terminal_recovery(
        stage_root=fixture.stage,
        planning_child_recovery_capability=capability,
    )
    assert detection["status"] == "valid_preterminal_recovery", detection


def test_r5_issuer_accepts_valid_preterminal_for_existing_workflow_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    import test_stage6_workflow as workflow_fixtures
    from lunar_exploration_ppo.utils import resource_lifecycle

    def append_historical_resource_segment(
        target_fixture,
        *,
        expected_index: int,
    ) -> dict[str, object]:
        historical_pid = os.getpid() + 1_000_000
        with monkeypatch.context() as historical_process:
            historical_process.setattr(
                resource_lifecycle,
                "_current_pid",
                lambda: historical_pid,
            )
            row = resource_lifecycle.append_resource_segment_start(
                target_fixture.stage / "resource_audit.jsonl",
                first_sample={
                    "rss_source": (
                        "process_tree_lifecycle_peak_current_sum/v1"
                    ),
                    "rss_root_pid": historical_pid,
                    "rss_sample_count": 1,
                    "rss_bytes": 1300,
                    "passed": True,
                },
            )
        assert row["segment_index"] == expected_index
        return row

    monkeypatch.setattr(
        workflow_fixtures,
        "_append_task5_resource_segment",
        append_historical_resource_segment,
    )
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    _prepare_u100_preterminal_evidence(
        fixture,
        capability,
        identity=identity,
        review=review,
        immutable=immutable,
        monkeypatch=monkeypatch,
        authority_root=tmp_path / "r5-preterminal-authority",
    )

    restarted = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert restarted.terminal_complete is False
    assert restarted.resume_cursor is None
    checkpoint_bytes = {
        path.relative_to(fixture.stage).as_posix(): path.read_bytes()
        for path in (fixture.stage / "checkpoints").rglob("*")
        if path.is_file()
    }
    from lunar_exploration_ppo.utils import resources
    from test_stage6_workflow import _active_execution_capability

    class FakeMonitor:
        def __init__(
            self,
            *,
            interval_seconds: float,
            **_kwargs: object,
        ) -> None:
            assert interval_seconds >= 3600.0
            self.running = False
            self.root_pid = os.getpid()
            self.sample_count = 0
            self.latest_process_count = 1
            self.peak_process_count = 1
            self.peak_aggregate_rss_bytes = 1024

        def start(self):
            self.running = True
            self.sample_count = 1
            return self

        def sample_now(self):
            assert self.running is True
            self.sample_count = 2
            self.peak_aggregate_rss_bytes = 2048
            return object()

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(resources, "ProcessTreeRSSMonitor", FakeMonitor)
    continuation_record = json.loads(
        continuation_path.read_text(encoding="utf-8")
    )
    review_authorization_path = Path(
        str(continuation_record["current"]["authorization"]["path"])
    )
    with _active_execution_capability(
        tmp_path / "r5-preterminal-convergence-authority",
        monkeypatch,
        run_root=fixture.stage.parent,
        effective_config_bytes=fixture.effective_config_bytes,
        planning_warm_start_path=fixture.warm_start_path,
        planning_child_source_repair_path=fixture.child_path,
        execution_identity_override=identity,
        verified_review_override=review,
        review_authorization_path=review_authorization_path,
        formal_run_id=fixture.stage.parent.name,
    ) as active:
        convergence = fixture.module._recover_stage6_preterminal_receipt(
            stage_root=fixture.stage,
            repo_root=ROOT,
            review_authorization_handle=active["review_handle"],
            input_pin=active["pin"],
            execution_capability=active["capability"],
            planning_child_recovery_capability=restarted,
        )

    assert convergence["phase_states"]
    detection = fixture.terminal_recovery.detect_stage6_terminal_recovery(
        stage_root=fixture.stage,
        manifest_verifier=lambda candidate: (
            fixture.module._verify_existing_stage6_manifest_identity(
                candidate,
                repo_root=ROOT,
                planning_child_recovery_capability=restarted,
            )
        ),
        planning_child_recovery_capability=restarted,
    )
    assert detection["status"] == "valid_terminal_recovery", detection
    assert checkpoint_bytes == {
        path.relative_to(fixture.stage).as_posix(): path.read_bytes()
        for path in (fixture.stage / "checkpoints").rglob("*")
        if path.is_file()
    }
    assert not any(
        path.name.startswith("update-00000101")
        for path in (fixture.stage / "checkpoints").rglob("*")
    )


def test_r5_partial_terminal_restart_reissues_capability_and_converges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    import test_stage6_workflow as workflow_fixtures
    from lunar_exploration_ppo.utils import resource_lifecycle
    from test_stage6_workflow import _active_execution_capability

    def append_historical_resource_segment(
        target_fixture,
        *,
        expected_index: int,
    ) -> dict[str, object]:
        historical_pid = os.getpid() + 2_000_000
        with monkeypatch.context() as historical_process:
            historical_process.setattr(
                resource_lifecycle,
                "_current_pid",
                lambda: historical_pid,
            )
            row = resource_lifecycle.append_resource_segment_start(
                target_fixture.stage / "resource_audit.jsonl",
                first_sample={
                    "rss_source": (
                        "process_tree_lifecycle_peak_current_sum/v1"
                    ),
                    "rss_root_pid": historical_pid,
                    "rss_sample_count": 1,
                    "rss_bytes": 1300,
                    "passed": True,
                },
            )
        assert row["segment_index"] == expected_index
        return row

    monkeypatch.setattr(
        workflow_fixtures,
        "_append_task5_resource_segment",
        append_historical_resource_segment,
    )
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    _prepare_u100_preterminal_evidence(
        fixture,
        capability,
        identity=identity,
        review=review,
        immutable=immutable,
        monkeypatch=monkeypatch,
        authority_root=tmp_path / "r5-partial-terminal-preterminal-authority",
    )
    checkpoint_bytes = {
        path.relative_to(fixture.stage).as_posix(): path.read_bytes()
        for path in (fixture.stage / "checkpoints").rglob("*")
        if path.is_file()
    }
    continuation_record = json.loads(
        continuation_path.read_text(encoding="utf-8")
    )
    review_authorization_path = Path(
        str(continuation_record["current"]["authorization"]["path"])
    )

    with _active_execution_capability(
        tmp_path / "r5-partial-terminal-append-authority",
        monkeypatch,
        run_root=fixture.stage.parent,
        effective_config_bytes=fixture.effective_config_bytes,
        planning_warm_start_path=fixture.warm_start_path,
        planning_child_source_repair_path=fixture.child_path,
        execution_identity_override=identity,
        verified_review_override=review,
        review_authorization_path=review_authorization_path,
        formal_run_id=fixture.stage.parent.name,
    ) as active:
        fixture.terminal_recovery.append_stage6_recovery_resource_segment(
            stage_root=fixture.stage,
            first_sample=_task5_terminal_resource(
                sample_count=1,
                rss_bytes=2048,
            ),
            execution_capability=active["capability"],
            planning_child_recovery_capability=capability,
        )
        fixture.terminal_recovery.append_stage6_recovery_resource_terminal(
            stage_root=fixture.stage,
            terminal_resource=_task5_terminal_resource(
                sample_count=2,
                rss_bytes=4096,
            ),
            execution_capability=active["capability"],
            planning_child_recovery_capability=capability,
        )

    partial = fixture.terminal_recovery.detect_stage6_terminal_recovery(
        stage_root=fixture.stage,
        planning_child_recovery_capability=capability,
    )
    assert partial["status"] == "valid_terminal_recovery", partial
    assert fixture.terminal_recovery.MANIFEST_NAME not in {
        path.name for path in fixture.stage.iterdir()
    }

    restarted = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    assert restarted.terminal_complete is False
    assert restarted.resume_cursor is None

    with _active_execution_capability(
        tmp_path / "r5-partial-terminal-convergence-authority",
        monkeypatch,
        run_root=fixture.stage.parent,
        effective_config_bytes=fixture.effective_config_bytes,
        planning_warm_start_path=fixture.warm_start_path,
        planning_child_source_repair_path=fixture.child_path,
        execution_identity_override=identity,
        verified_review_override=review,
        review_authorization_path=review_authorization_path,
        formal_run_id=fixture.stage.parent.name,
    ) as active:
        convergence = fixture.terminal_recovery.recover_stage6_terminal_commit(
            stage_root=fixture.stage,
            manifest_committer=lambda candidate: (
                fixture.module.write_or_verify_stage6_manifest(
                    candidate,
                    repo_root=ROOT,
                    planning_child_recovery_capability=restarted,
                )
            ),
            execution_capability=active["capability"],
            planning_child_recovery_capability=restarted,
        )

    assert convergence["phase_states"]
    assert checkpoint_bytes == {
        path.relative_to(fixture.stage).as_posix(): path.read_bytes()
        for path in (fixture.stage / "checkpoints").rglob("*")
        if path.is_file()
    }
    assert not any(
        path.name.startswith("update-00000101")
        for path in (fixture.stage / "checkpoints").rglob("*")
    )


def test_r5_terminal_receipt_accepts_u86_start_capability_as_historical_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    u86_capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    assert u86_capability.resume_cursor is not None
    assert u86_capability.resume_cursor.next_update == 86
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    _prepare_u100_terminal_evidence(
        fixture,
        u86_capability,
        identity=identity,
        review=review,
        immutable=immutable,
        monkeypatch=monkeypatch,
    )

    restarted = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert restarted.terminal_complete is True
    assert restarted.resume_cursor is None
    assert restarted.acceptance_binding["resume_cursor"]["next_update"] == 86
    assert restarted.acceptance_binding["accepted_anchor"][
        "last_accepted_update"
    ] == 85
    assert restarted.protected_checkpoint_updates == (
        u86_capability.protected_checkpoint_updates
    )


def test_recovery_anchor_is_preview_only_and_cannot_launch_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        _continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)

    anchor = recovery.inspect_planning_child_recovery_anchor(
        stage_root=fixture.stage,
        parent_artifact_path=fixture.child_path,
    )

    assert isinstance(anchor, recovery.PlanningChildRecoveryAnchor)
    assert not isinstance(
        anchor,
        recovery.PlanningChildRecoveryCapability,
    )
    assert not hasattr(anchor, "resume_cursor")
    with pytest.raises(
        recovery.Stage6PlanningChildRecoveryError,
        match="continuation",
    ):
        recovery.issue_planning_child_recovery_capability(
            stage_root=fixture.stage,
            parent_artifact_path=fixture.child_path,
            continuation_artifact_path=None,
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
        )


def test_recovery_capability_derives_initial_u86_attempt1_segment5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert _cursor_tuple(capability) == (
        85,
        86,
        1,
        "0011:20260716:update:086",
        5,
        85,
        None,
    )


def test_recovery_capability_finalizes_u100_without_u101_then_passes_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert capability.terminal_complete is False
    assert capability.resume_cursor is None
    assert capability.acceptance_binding["terminal_complete"] is False
    assert capability.acceptance_binding["resume_cursor"] is None
    assert capability.acceptance_binding["accepted_anchor"][
        "last_accepted_update"
    ] == 100
    assert tuple(
        (epoch.first_update, epoch.last_update)
        for epoch in capability.lineage_epochs
    ) == ((75, 84), (85, 85), (86, None))

    _prepare_u100_terminal_evidence(
        fixture,
        capability,
        identity=identity,
        review=review,
        immutable=immutable,
        monkeypatch=monkeypatch,
    )

    terminal_capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    assert terminal_capability.terminal_complete is True
    assert terminal_capability.resume_cursor is None

    real_detect = (
        fixture.terminal_recovery.detect_stage6_terminal_recovery
    )
    detections: list[dict[str, object]] = []

    def detect(**kwargs: object) -> dict[str, object]:
        result = real_detect(**kwargs)
        detections.append(dict(result))
        return result

    monkeypatch.setattr(
        fixture.terminal_recovery,
        "detect_stage6_terminal_recovery",
        detect,
    )
    machine = fixture.module.verify_stage6_machine_acceptance(
        stage_root=fixture.stage,
        repo_root=ROOT,
        planning_child_recovery_capability=terminal_capability,
    )

    assert machine["passed"] is True
    assert detections[-1]["status"] == "valid_terminal_recovery"


def test_recovery_capability_rejects_partial_u100_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    _prepare_u100_terminal_evidence(
        fixture,
        capability,
        identity=identity,
        review=review,
        immutable=immutable,
        monkeypatch=monkeypatch,
    )
    (fixture.stage / "routing.json").unlink()

    with pytest.raises(
        recovery.Stage6PlanningChildRecoveryError,
        match="terminal|evidence|recovery",
    ):
        _issue(
            fixture,
            continuation_path,
            identity=identity,
            review=review,
            immutable=immutable,
        )


@pytest.mark.parametrize(
    "boundary",
    ("accepted", "checkpoint", "receipt", "journal"),
)
def test_recovery_capability_keeps_u100_crash_for_backend_convergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    prefix = _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=99,
    )
    _append_task5_u86_crash_suffix(
        fixture,
        tail,
        boundary=boundary,
        transaction_update=100,
        segment=prefix.segment,
        prior_policy_state_sha256=(
            prefix.checkpoint.policy_state_sha256
        ),
        resource_sample_count=30,
        resource_rss_bytes=2800,
    )

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert capability.terminal_complete is False
    assert capability.resume_cursor is None
    assert capability.acceptance_binding["terminal_complete"] is False
    assert capability.acceptance_binding["resume_cursor"] is None
    assert capability.acceptance_binding["crash_suffix"][
        "boundary"
    ] == boundary
    assert capability.acceptance_binding["crash_suffix"][
        "update"
    ] == 100


def test_recovery_capability_marks_u100_training_complete_finalize_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert capability.terminal_complete is False
    assert capability.resume_cursor is None
    assert capability.acceptance_binding["terminal_complete"] is False
    assert capability.acceptance_binding["resume_cursor"] is None
    assert capability.acceptance_binding["crash_suffix"] is None


@pytest.mark.parametrize("orphan_kind", ("complete", "pending"))
def test_recovery_capability_rejects_u101_orphan_checkpoint_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    orphan_kind: str,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    completed = _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=100,
    )
    checkpoint_parent = completed.checkpoint_root.parent
    if orphan_kind == "complete":
        orphan = checkpoint_parent / "update-00000101"
        orphan.mkdir()
        for name in ("checkpoint.pt", "manifest.json", "complete.json"):
            (orphan / name).write_bytes(
                (completed.checkpoint_root / name).read_bytes()
            )
    else:
        orphan = (
            checkpoint_parent
            / ".pending-update-00000101-22222222222222222222222222222222"
        )
        orphan.mkdir()
        (orphan / "checkpoint.pt").write_bytes(b"pending-u101")

    with pytest.raises(
        recovery.Stage6PlanningChildRecoveryError,
        match="checkpoint|U101|ahead|orphan",
    ):
        _issue(
            fixture,
            continuation_path,
            identity=identity,
            review=review,
            immutable=immutable,
        )


@pytest.mark.parametrize(
    "boundary",
    ("accepted", "checkpoint", "receipt", "journal", "training"),
)
def test_u100_recovery_reuses_active_segment_and_converges_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from contextlib import nullcontext

    import torch

    from lunar_exploration_ppo.ppo import (
        checkpoint as checkpoint_module,
        standard_training,
    )

    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=(84, 90),
    )
    prefix = _append_task5_accepted_through_u90_restart_tail(
        fixture,
        tail,
        final_update=99,
    )
    suffix = _append_task5_u86_crash_suffix(
        fixture,
        tail,
        boundary=boundary,
        transaction_update=100,
        segment=prefix.segment,
        prior_policy_state_sha256=(
            prefix.checkpoint.policy_state_sha256
        ),
        resource_sample_count=30,
        resource_rss_bytes=2800,
    )
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    resource_path = fixture.stage / "resource_audit.jsonl"
    resource_before = resource_path.read_bytes()
    segment_start = (
        standard_training._append_initial_resource_segment_start(
            stage_root=fixture.stage,
            first_sample={
                "rss_source": (
                    "process_tree_lifecycle_peak_current_sum/v1"
                ),
                "rss_root_pid": int(prefix.segment["root_pid"]),
                "rss_sample_count": 32,
                "rss_bytes": 2900,
                "passed": True,
            },
            planning_child_recovery_capability=capability,
        )
    )
    assert segment_start == prefix.segment
    assert resource_path.read_bytes() == resource_before

    class RestorableTrainer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.update_step = 0

        def restore_update_step(self, update_step: int) -> None:
            self.update_step = update_step

    class RecoveryReachedRuntimeBoundary(RuntimeError):
        pass

    backend = object.__new__(
        standard_training.StandardProductionBackend
    )
    backend._execution_operation = lambda _label: nullcontext()
    backend._require_execution_capability_current = (
        lambda _label, *, rehash_inputs=False: None
    )
    backend._planning_child_recovery_capability = capability
    backend._planning_child_resume_boundary_consumed = False
    backend._planning_warm_start = object()
    backend._planning_warm_start_sha256 = "1" * 64
    backend._source_repair = None
    backend._immutable_bindings = dict(immutable)
    backend.config = fixture.config
    backend.config_sha256 = str(immutable["config_sha256"])
    backend.safety_contract = fixture.safety_contract
    backend.stage_root = fixture.stage
    backend.run_root = fixture.stage.parent
    backend.repo_root = ROOT
    backend.stage5_authority = dict(fixture.authority.identity)
    backend.transactions = tuple(fixture.transactions)
    backend.receipt_index = standard_training.CheckpointReceiptIndex(
        fixture.stage / "checkpoints/index.jsonl"
    )
    backend.journal = fixture.module.Stage6StateJournal(
        fixture.stage / "job-state.jsonl"
    )
    backend._resource_segment_start = segment_start
    backend._components = {
        "CheckpointManager": checkpoint_module.CheckpointManager,
        "stage4_checkpoint_path": (
            fixture.stage / "fixture-stage4-checkpoint.pt"
        ),
        "load_stage4_policy_for_standard": (
            lambda **_kwargs: torch.nn.Linear(1, 1)
        ),
        "PPOTrainer": RestorableTrainer,
    }
    backend._apply_checkpoint_retention = (
        lambda **_kwargs: None
    )
    observed_remaining: list[tuple[str, ...]] = []
    real_resume_boundary = (
        standard_training.StandardProductionBackend
        ._require_planning_child_resume_boundary.__get__(backend)
    )

    def observe_resume_boundary(remaining: object) -> None:
        observed_remaining.append(
            tuple(item.key for item in remaining)
        )
        real_resume_boundary(remaining)

    backend._require_planning_child_resume_boundary = (
        observe_resume_boundary
    )

    def stop_after_recovery(**_kwargs: object) -> object:
        raise RecoveryReachedRuntimeBoundary

    backend._production_training_env_specs = stop_after_recovery
    seed_transactions = tuple(
        transaction
        for transaction in fixture.transactions
        if transaction.seed == fixture.transactions[0].seed
    )
    with pytest.raises(RecoveryReachedRuntimeBoundary):
        backend.prepare_seed(
            fixture.transactions[0].seed,
            seed_transactions,
        )

    assert observed_remaining == [()]
    receipts = backend.receipt_index.verify()
    completed = standard_training.verify_journal_checkpoint_bindings(
        backend.journal.verify(),
        backend.transactions,
        receipts,
        immutable,
        immutable_binding_epochs=backend._journal_binding_epochs(),
    )
    assert len(receipts) == len(fixture.transactions) == 26
    assert len(completed) == len(fixture.transactions)
    assert len(
        (fixture.stage / "training_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == len(fixture.transactions)
    resource_rows = [
        json.loads(line)
        for line in resource_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert sum(
        row.get("phase") == "segment_start"
        for row in resource_rows
    ) == int(prefix.segment["segment_index"])
    assert not any(
        path.name.startswith(
            ("update-00000101", ".pending-update-00000101-")
        )
        for path in suffix.checkpoint_root.parent.iterdir()
    )


def test_recovery_capability_derives_segment5_start_as_u86_attempt1_segment6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    _append_task5_resource_segment(fixture, expected_index=5)

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert _cursor_tuple(capability) == (
        85,
        86,
        1,
        "0011:20260716:update:086",
        6,
        85,
        None,
    )


def test_recovery_capability_derives_segment5_pre_as_u86_attempt2_segment6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    segment = _append_task5_resource_segment(
        fixture,
        expected_index=5,
    )
    _append_task5_resource_pre(
        fixture,
        segment=segment,
        transaction=fixture.transactions[11],
        attempt=1,
    )

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert _cursor_tuple(capability) == (
        85,
        86,
        2,
        "0011:20260716:update:086",
        6,
        85,
        1,
    )


@pytest.mark.parametrize(
    "boundary",
    (
        "post",
        "accepted",
        "checkpoint",
        "receipt",
        "journal",
        "training",
    ),
)
def test_recovery_capability_classifies_one_u86_production_crash_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    suffix = _append_task5_u86_crash_suffix(
        fixture,
        tail,
        boundary=boundary,
    )

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    cursor = capability.resume_cursor
    assert cursor is not None
    if boundary == "post":
        assert (
            cursor.next_update,
            cursor.next_attempt,
            cursor.next_transaction_key,
        ) == (86, 2, "0011:20260716:update:086")
    else:
        assert (
            cursor.next_update,
            cursor.next_attempt,
            cursor.next_transaction_key,
        ) == (87, 1, "0012:20260716:update:087")
    assert cursor.next_resource_segment_index == 6
    if boundary == "training":
        assert cursor.last_accepted_update == 86
        assert cursor.latest_complete_checkpoint_update == 86
        assert capability.acceptance_binding["crash_suffix"] is None
        return
    crash = capability.acceptance_binding["crash_suffix"]
    assert crash["schema_version"] == (
        "stage6_planning_child_recovery_crash_suffix/v1"
    )
    assert crash["boundary"] == boundary
    assert crash["transaction_key"] == (
        "0011:20260716:update:086"
    )
    assert crash["attempt"] == 1
    if boundary in {
        "accepted",
        "checkpoint",
        "receipt",
        "journal",
    }:
        assert crash["checkpoint"]["checkpoint_sha256"] == (
            suffix.checkpoint.checkpoint_sha256
        )
    else:
        assert crash["checkpoint"] is None


@pytest.mark.parametrize(
    "mutation",
    (
        "more_than_one_ahead",
        "checkpoint_mismatch",
        "reorder",
        "duplicate",
        "unrelated_suffix",
    ),
)
def test_recovery_capability_rejects_invalid_u86_crash_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    boundary = (
        "accepted"
        if mutation
        in {"more_than_one_ahead", "checkpoint_mismatch"}
        else ("pre" if mutation == "unrelated_suffix" else "post")
    )
    suffix = _append_task5_u86_crash_suffix(
        fixture,
        tail,
        boundary=boundary,
    )
    resource_path = fixture.stage / "resource_audit.jsonl"
    rows = [
        json.loads(line)
        for line in resource_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    if mutation == "more_than_one_ahead":
        _append_task5_resource_pre(
            fixture,
            segment=suffix.segment,
            transaction=fixture.transactions[12],
            attempt=1,
        )
    elif mutation == "checkpoint_mismatch":
        rows[-1]["checkpoint"]["checkpoint_sha256"] = "0" * 64
    elif mutation == "reorder":
        rows[-2], rows[-1] = rows[-1], rows[-2]
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif mutation == "unrelated_suffix":
        unrelated = fixture.transactions[12]
        rows[-1].update(
            {
                "transaction_key": unrelated.key,
                "seed": unrelated.seed,
                "update": unrelated.update,
            }
        )
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    if mutation != "more_than_one_ahead":
        resource_path.write_bytes(
            b"".join(
                fixture.module.ArtifactStore.canonical_json_bytes(row)
                for row in rows
            )
        )

    with pytest.raises(recovery.Stage6PlanningChildRecoveryError):
        _issue(
            fixture,
            continuation_path,
            identity=identity,
            review=review,
            immutable=immutable,
        )


@pytest.mark.parametrize(
    "boundary",
    (
        "post",
        "accepted",
        "checkpoint",
        "receipt",
        "journal",
        "training",
    ),
)
def test_recovery_capability_suffix_converges_through_production_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from contextlib import nullcontext

    import torch

    from lunar_exploration_ppo.ppo import (
        checkpoint as checkpoint_module,
        standard_training,
    )

    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    _append_task5_u86_crash_suffix(
        fixture,
        tail,
        boundary=boundary,
    )
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    cursor = capability.resume_cursor
    assert cursor is not None

    class RestorableTrainer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.update_step = 0

        def restore_update_step(self, update_step: int) -> None:
            self.update_step = update_step

    class RecoveryReachedRuntimeBoundary(RuntimeError):
        pass

    backend = object.__new__(
        standard_training.StandardProductionBackend
    )
    backend._execution_operation = lambda _label: nullcontext()
    backend._require_execution_capability_current = (
        lambda _label, *, rehash_inputs=False: None
    )
    backend._planning_child_recovery_capability = capability
    backend._planning_child_resume_boundary_consumed = False
    backend._planning_warm_start = object()
    backend._planning_warm_start_sha256 = "1" * 64
    backend._source_repair = None
    backend._immutable_bindings = dict(immutable)
    backend.config = fixture.config
    backend.config_sha256 = str(immutable["config_sha256"])
    backend.safety_contract = fixture.safety_contract
    backend.stage_root = fixture.stage
    backend.run_root = fixture.stage.parent
    backend.repo_root = ROOT
    backend.stage5_authority = dict(fixture.authority.identity)
    backend.transactions = tuple(fixture.transactions)
    backend.receipt_index = standard_training.CheckpointReceiptIndex(
        fixture.stage / "checkpoints/index.jsonl"
    )
    backend.journal = fixture.module.Stage6StateJournal(
        fixture.stage / "job-state.jsonl"
    )
    backend._resource_segment_start = {
        "segment_index": cursor.next_resource_segment_index
    }
    backend._components = {
        "CheckpointManager": checkpoint_module.CheckpointManager,
        "stage4_checkpoint_path": (
            fixture.stage / "fixture-stage4-checkpoint.pt"
        ),
        "load_stage4_policy_for_standard": (
            lambda **_kwargs: torch.nn.Linear(1, 1)
        ),
        "PPOTrainer": RestorableTrainer,
    }
    backend._apply_checkpoint_retention = (
        lambda **_kwargs: None
    )
    observed_remaining: list[tuple[str, ...]] = []
    real_resume_boundary = (
        standard_training.StandardProductionBackend
        ._require_planning_child_resume_boundary.__get__(backend)
    )

    def observe_resume_boundary(
        remaining: object,
    ) -> None:
        observed_remaining.append(
            tuple(item.key for item in remaining)
        )
        real_resume_boundary(remaining)

    backend._require_planning_child_resume_boundary = (
        observe_resume_boundary
    )

    def stop_after_recovery(**_kwargs: object) -> object:
        raise RecoveryReachedRuntimeBoundary

    backend._production_training_env_specs = stop_after_recovery

    seed_transactions = tuple(
        transaction
        for transaction in fixture.transactions
        if transaction.seed == fixture.transactions[0].seed
    )
    with pytest.raises(RecoveryReachedRuntimeBoundary):
        backend.prepare_seed(
            fixture.transactions[0].seed,
            seed_transactions,
        )

    expected_first = (
        "0011:20260716:update:086"
        if boundary == "post"
        else "0012:20260716:update:087"
    )
    assert observed_remaining
    assert observed_remaining[-1][0] == expected_first
    receipts = backend.receipt_index.verify()
    completed = standard_training.verify_journal_checkpoint_bindings(
        backend.journal.verify(),
        backend.transactions,
        receipts,
        immutable,
        immutable_binding_epochs=(
            backend._journal_binding_epochs()
        ),
    )
    training_rows = (
        fixture.stage / "training_metrics.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    expected_count = 11 if boundary == "post" else 12
    assert len(receipts) == expected_count
    assert len(completed) == expected_count
    assert len(training_rows) == expected_count
    if boundary == "accepted":
        assert not any(
            path.name.startswith(".pending-update-00000086-")
            for path in (
                fixture.stage / "checkpoints/seed-20260716"
            ).iterdir()
        )


def test_recovery_capability_derives_u86_accepted_as_u87_attempt1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    _append_task5_accepted_u86_restart_tail(fixture, tail)

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert _cursor_tuple(capability) == (
        86,
        87,
        1,
        "0012:20260716:update:087",
        6,
        86,
        None,
    )


def _mutate_recovery_journal(
    fixture,
    mutation: str,
) -> None:
    if mutation == "rewrite":
        path = fixture.stage / "training_metrics.jsonl"
        payload = path.read_bytes()
        path.write_bytes(b"[" + payload[1:])
        return
    if mutation == "truncation":
        path = fixture.stage / "resource_audit.jsonl"
        path.write_bytes(path.read_bytes()[:-1])
        return
    if mutation == "reorder":
        path = fixture.stage / "job-state.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        lines[-2], lines[-1] = lines[-1], lines[-2]
        path.write_bytes(b"".join(lines))
        return
    if mutation == "duplicate":
        path = fixture.stage / "resource_audit.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(path.read_bytes() + lines[-1])
        return
    if mutation == "gap":
        path = fixture.stage / "training_metrics.jsonl"
        row = json.loads(
            path.read_bytes().splitlines()[-1].decode("utf-8")
        )
        row.update(
            {
                "transaction_key": "0012:20260716:update:087",
                "update": 87,
            }
        )
        path.write_bytes(
            path.read_bytes()
            + fixture.module.ArtifactStore.canonical_json_bytes(row)
        )
        return
    raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    ("rewrite", "truncation", "reorder", "duplicate", "gap"),
)
def test_recovery_capability_rejects_rewrite_truncation_reorder_duplicate_and_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    _mutate_recovery_journal(fixture, mutation)

    with pytest.raises(recovery.Stage6PlanningChildRecoveryError):
        _issue(
            fixture,
            continuation_path,
            identity=identity,
            review=review,
            immutable=immutable,
        )


@pytest.mark.parametrize(
    "drift",
    ("identity", "authorization", "artifact", "checkpoint"),
)
def test_recovery_capability_rejects_identity_authorization_artifact_and_checkpoint_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    identity = copy.deepcopy(identity)
    review = copy.deepcopy(review)
    immutable = copy.deepcopy(immutable)
    if drift == "identity":
        identity["source_set_sha256"] = "0" * 64
    elif drift == "authorization":
        review["authorized"] = False
    elif drift == "artifact":
        continuation_path.write_bytes(
            continuation_path.read_bytes() + b"\n"
        )
    elif drift == "checkpoint":
        path = (
            fixture.stage
            / "checkpoints"
            / "seed-20260716"
            / "update-00000085"
            / "checkpoint.pt"
        )
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(bytes(payload))
    else:
        raise AssertionError(f"unknown drift: {drift}")

    with pytest.raises(recovery.Stage6PlanningChildRecoveryError):
        _issue(
            fixture,
            continuation_path,
            identity=identity,
            review=review,
            immutable=immutable,
        )


def test_recovery_capability_uses_one_bound_snapshot_and_methods_do_no_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path / "issued", monkeypatch)
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    def forbidden_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("capability method reopened recovery input")

    monkeypatch.setattr(recovery, "secure_read_bytes", forbidden_read)
    lineage = capability.checkpoint_lineage_for_update(86)
    assert lineage["checkpoint_update"] == 86
    assert lineage["planning_child_source_repair"][
        "artifact_sha256"
    ] == capability.lineage_epochs[2].artifact_sha256
    assert capability.resume_cursor.next_update == 86
    assert capability.acceptance_binding["resume_cursor"][
        "next_update"
    ] == 86
    assert capability.protected_checkpoint_updates == (84, 85)
    assert capability.input_pin_requests

    monkeypatch.undo()
    (
        drift_fixture,
        _drift_tail,
        drift_identity,
        drift_review,
        drift_immutable,
        drift_continuation_path,
    ) = _published_chain(tmp_path / "drift", monkeypatch)
    real_inspector = recovery.inspect_complete_checkpoint_snapshot
    calls: list[int] = []

    def mutate_after_capture(**kwargs: object) -> object:
        calls.append(int(kwargs["update_step"]))
        result = real_inspector(**kwargs)
        path = drift_fixture.stage / "job-state.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(
        recovery,
        "inspect_complete_checkpoint_snapshot",
        mutate_after_capture,
    )
    with pytest.raises(
        recovery.Stage6PlanningChildRecoveryError,
        match="changed",
    ):
        _issue(
            drift_fixture,
            drift_continuation_path,
            identity=drift_identity,
            review=drift_review,
            immutable=drift_immutable,
        )
    assert calls == [85]


def test_recovery_capability_deep_freezes_nested_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)
    identity = copy.deepcopy(identity)
    review = copy.deepcopy(review)
    immutable = copy.deepcopy(immutable)
    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )
    digest = capability.capability_sha256
    lineage = capability.checkpoint_lineage_for_update(86)

    identity["source_set_sha256"] = "0" * 64
    review["authorized"] = False
    environment = immutable["environment_identity"]
    assert isinstance(environment, dict)
    environment["python_version"] = "caller-mutated"

    with pytest.raises(TypeError):
        capability.acceptance_binding["resume_cursor"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        capability.acceptance_binding["lineage_epochs"][0][  # type: ignore[index]
            "first_update"
        ] = 0

    assert capability.capability_sha256 == digest
    assert capability.checkpoint_lineage_for_update(86) == lineage
    assert capability.acceptance_binding["resume_cursor"][
        "next_update"
    ] == 86


def test_recovery_capability_derives_three_lineage_epochs_and_pins_u84_u85(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        continuation_path,
    ) = _published_chain(tmp_path, monkeypatch)

    capability = _issue(
        fixture,
        continuation_path,
        identity=identity,
        review=review,
        immutable=immutable,
    )

    assert tuple(
        (epoch.first_update, epoch.last_update)
        for epoch in capability.lineage_epochs
    ) == ((75, 84), (85, 85), (86, None))
    assert capability.protected_checkpoint_updates == (84, 85)
    origin_lineage = capability.checkpoint_lineage_for_update(84)
    parent_lineage = capability.checkpoint_lineage_for_update(85)
    continuation_lineage = capability.checkpoint_lineage_for_update(86)
    assert origin_lineage["checkpoint_update"] == 84
    assert "planning_child_source_repair" not in origin_lineage
    assert parent_lineage["planning_child_source_repair"][
        "artifact_sha256"
    ] == capability.lineage_epochs[1].artifact_sha256
    assert continuation_lineage["planning_child_source_repair"][
        "artifact_sha256"
    ] == capability.lineage_epochs[2].artifact_sha256
