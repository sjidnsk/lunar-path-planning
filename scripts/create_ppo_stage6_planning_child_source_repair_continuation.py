"""Preview or publish the write-once Stage 6 planning-child continuation."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.durable_jsonl import RunLease
from lunar_exploration_ppo.utils.path_security import secure_read_bytes
from lunar_exploration_ppo.workflows.stage6 import (
    CANONICAL_STAGE5_GATE,
    _stage6_current_immutable_bindings,
    stage6_execution_identity,
    validate_stage6_verified_review_authorization,
    verify_frozen_stage5_authority,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair_continuation import (
    PlanningChildContinuationPublishCapability,
    _require_continuation_run_lease,
    build_planning_child_source_repair_continuation_artifact,
    issue_planning_child_source_repair_continuation_publish_capability,
    publish_planning_child_source_repair_continuation_artifact,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
    inspect_planning_child_recovery_anchor,
)
from lunar_exploration_ppo.workflows.stage6_review_authorization import (
    verify_stage6_review_launch_authorization,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    DEFAULT_REPO_ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
)


def _created_at_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def create_planning_child_source_repair_continuation_preview(
    *,
    stage_root: Path,
    run_id: str,
    parent_path: Path,
    review_authorization_path: Path,
    implementation_report_path: Path,
    spec_review_path: Path,
    quality_review_path: Path,
    run_lease: RunLease,
    stage5_gate_path: Path = CANONICAL_STAGE5_GATE,
) -> PlanningChildContinuationPublishCapability:
    """Build a continuation preview without writing the stage root."""

    lease = _require_continuation_run_lease(
        stage_root=stage_root,
        run_lease=run_lease,
        label="planning child continuation preview",
    )
    effective_config_bytes = secure_read_bytes(
        stage_root / "config.json",
        base=stage_root,
        label="planning child continuation effective config",
    ).payload
    authorization = verify_stage6_review_launch_authorization(
        authorization_path=review_authorization_path,
        repo_root=DEFAULT_REPO_ROOT,
        config_path=DEFAULT_CONFIG_PATH,
        expected_run_id=run_id,
        effective_config_bytes=effective_config_bytes,
    )
    authorization.require_current(
        "planning child continuation preview entry"
    )
    identity = stage6_execution_identity(
        repo_root=DEFAULT_REPO_ROOT,
        config_path=DEFAULT_CONFIG_PATH,
        effective_config_bytes=effective_config_bytes,
    )
    review = validate_stage6_verified_review_authorization(
        authorization.canonical_record(),
        execution_identity=identity,
        formal_run_id=run_id,
    )
    authority = verify_frozen_stage5_authority(
        gate_path=stage5_gate_path,
        repo_root=DEFAULT_REPO_ROOT,
    )
    immutable = _stage6_current_immutable_bindings(
        execution_identity=identity,
        verified_review_authorization=review,
        stage5_authority=authority,
    )
    anchor = inspect_planning_child_recovery_anchor(
        stage_root=stage_root,
        parent_artifact_path=parent_path,
    )
    artifact = (
        build_planning_child_source_repair_continuation_artifact(
            stage_root=stage_root,
            parent_path=parent_path,
            anchor=anchor,
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
            current_review_authorization_path=(
                review_authorization_path
            ),
            implementation_report_path=implementation_report_path,
            spec_review_path=spec_review_path,
            quality_review_path=quality_review_path,
            created_at_utc=_created_at_utc(),
        )
    )
    authority.require_current(
        "planning child continuation preview exit"
    )
    authorization.require_current(
        "planning child continuation preview exit"
    )
    lease.require_current()
    return issue_planning_child_source_repair_continuation_publish_capability(
        artifact,
        stage_root=stage_root,
        anchor=anchor,
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
        authorization_currentness=authorization,
        stage5_authority_currentness=authority,
        effective_config_bytes=effective_config_bytes,
        run_lease=lease,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the Stage 6 planning-child continuation. No stage "
            "artifact is written unless --publish is supplied."
        )
    )
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument(
        "--review-authorization",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--implementation-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--spec-review", required=True, type=Path)
    parser.add_argument("--quality-review", required=True, type=Path)
    parser.add_argument(
        "--stage5-gate",
        type=Path,
        default=CANONICAL_STAGE5_GATE,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Atomically publish the canonical artifact exactly once.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with RunLease(args.stage_root.parent / ".stage6.lease") as run_lease:
        artifact = create_planning_child_source_repair_continuation_preview(
            stage_root=args.stage_root,
            run_id=args.run_id,
            parent_path=args.parent,
            review_authorization_path=args.review_authorization,
            implementation_report_path=args.implementation_report,
            spec_review_path=args.spec_review,
            quality_review_path=args.quality_review,
            run_lease=run_lease,
            stage5_gate_path=args.stage5_gate,
        )
        if args.publish:
            publish_planning_child_source_repair_continuation_artifact(
                artifact,
                stage_root=args.stage_root,
                output_path=args.output,
                run_lease=run_lease,
            )
    sys.stdout.write(
        ArtifactStore.canonical_json_bytes(
            artifact.canonical_record()
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
