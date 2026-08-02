"""Preview or explicitly publish the Stage 6 planning-child repair bridge."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from lunar_exploration_ppo.configs.stage6 import load_stage6_config
from lunar_exploration_ppo.ppo.standard_training import (
    _review_authorization_immutable_bindings,
)
from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.utils.path_security import secure_read_bytes
from lunar_exploration_ppo.workflows.stage6 import (
    STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256,
    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256,
    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES,
    STAGE6_COVERAGE_CACHE_MANIFEST_PATH,
    STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
    STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES,
    STAGE6_COVERAGE_CACHE_ROOT,
    STAGE6_COVERAGE_CACHE_RUNTIME_MODE,
    stage6_execution_identity,
    validate_stage6_verified_review_authorization,
)
from lunar_exploration_ppo.workflows.stage6_planning_child_source_repair import (
    PLANNING_CHILD_SOURCE_REPAIR_NAME,
    build_planning_child_source_repair_artifact,
    publish_planning_child_source_repair_artifact,
)
from lunar_exploration_ppo.workflows.stage6_review_authorization import (
    verify_stage6_review_launch_authorization,
)


SEED = 20260716
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    DEFAULT_REPO_ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
)


def _created_at_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _current_immutable_bindings(
    *,
    identity: dict[str, object],
    review: dict[str, object],
) -> dict[str, object]:
    config = load_stage6_config(DEFAULT_CONFIG_PATH)
    result = {
        key: identity[key]
        for key in (
            "config_sha256",
            "source_set_sha256",
            "prospective_tree_sha256",
            "data_sha256",
            "environment_identity",
            "environment_sha256",
        )
    }
    result["stage5_gate_sha256"] = config.stage5_authority.gate_sha256
    result.update(_review_authorization_immutable_bindings(review))
    result.update(
        {
            "coverage_cache_manifest_path": (
                STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix()
            ),
            "coverage_cache_manifest_sha256": (
                STAGE6_COVERAGE_CACHE_MANIFEST_SHA256
            ),
            "coverage_cache_manifest_size_bytes": (
                STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
            ),
            "coverage_cache_root": STAGE6_COVERAGE_CACHE_ROOT.as_posix(),
            "coverage_cache_entry_set_sha256": (
                STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256
            ),
            "coverage_cache_runtime_mode": (
                STAGE6_COVERAGE_CACHE_RUNTIME_MODE
            ),
            "coverage_cache_formal_audit_sha256": (
                STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
            ),
            "coverage_cache_formal_audit_size_bytes": (
                STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
            ),
        }
    )
    return result


def create_planning_child_source_repair_preview(
    *,
    stage_root: Path,
    run_id: str,
    review_authorization_path: Path,
    planning_warm_start_path: Path,
    spec_review_path: Path,
    quality_review_path: Path,
    implementation_report_path: Path,
    exact_replay_evidence_path: Path,
) -> dict[str, object]:
    """Verify current authority and build a write-free amendment preview."""

    effective_config_bytes = secure_read_bytes(
        stage_root / "config.json",
        base=stage_root,
        label="planning child effective config",
    ).payload
    authorization = verify_stage6_review_launch_authorization(
        authorization_path=review_authorization_path,
        repo_root=DEFAULT_REPO_ROOT,
        config_path=DEFAULT_CONFIG_PATH,
        expected_run_id=run_id,
        effective_config_bytes=effective_config_bytes,
    )
    authorization.require_current("planning child source-repair preview entry")
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
    immutable = _current_immutable_bindings(
        identity=identity,
        review=review,
    )
    artifact = build_planning_child_source_repair_artifact(
        stage_root=stage_root,
        formal_run_id=run_id,
        seed=SEED,
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
        current_review_authorization_path=review_authorization_path,
        planning_warm_start_path=planning_warm_start_path,
        spec_review_path=spec_review_path,
        quality_review_path=quality_review_path,
        implementation_report_path=implementation_report_path,
        exact_replay_evidence_path=exact_replay_evidence_path,
        created_at_utc=_created_at_utc(),
    )
    authorization.require_current("planning child source-repair preview exit")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the reviewed Stage 6 planning-child source repair. "
            "No file is written unless --publish is supplied."
        )
    )
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--review-authorization", required=True, type=Path)
    parser.add_argument("--planning-warm-start", required=True, type=Path)
    parser.add_argument("--spec-review", required=True, type=Path)
    parser.add_argument("--quality-review", required=True, type=Path)
    parser.add_argument("--implementation-report", required=True, type=Path)
    parser.add_argument("--exact-replay-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Atomically publish the canonical artifact exactly once.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = create_planning_child_source_repair_preview(
        stage_root=args.stage_root,
        run_id=args.run_id,
        review_authorization_path=args.review_authorization,
        planning_warm_start_path=args.planning_warm_start,
        spec_review_path=args.spec_review,
        quality_review_path=args.quality_review,
        implementation_report_path=args.implementation_report,
        exact_replay_evidence_path=args.exact_replay_evidence,
    )
    if args.publish:
        publish_planning_child_source_repair_artifact(
            artifact,
            stage_root=args.stage_root,
            output_path=args.output,
        )
    sys.stdout.write(
        ArtifactStore.canonical_json_bytes(artifact).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
