"""Create/replay the immutable Stage 6A source-repair ordinal chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar_exploration_ppo.configs.stage6 import load_stage6_config
from lunar_exploration_ppo.ppo.standard_training import (
    _review_authorization_immutable_bindings,
)
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
from lunar_exploration_ppo.workflows.stage6_review_authorization import (
    verify_stage6_review_launch_authorization,
)
from lunar_exploration_ppo.workflows.stage6_source_repair import (
    FIXED_FORMAL_RUN_ID,
    Stage6SourceRepairContext,
    create_stage6_source_repair_amendment,
)


DEFAULT_STAGE_ROOT = Path(
    "D:/xunce/out/ppo_frontier/"
    f"{FIXED_FORMAL_RUN_ID}/s6"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or explicitly publish the fixed Stage 6A ordinal chain "
            "through the ordinal5 frontier recovery parent and the Update 50 "
            "ordinal6 sensor acceleration and exact coverable cache. Default "
            "mode is preview-only; pass --publish to publish the validated "
            "next ordinal."
        )
    )
    parser.add_argument("--review-authorization", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ppo_highres_frontier_stage6_v1.json"),
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    publication_mode = parser.add_mutually_exclusive_group()
    publication_mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Select preview-only mode explicitly; preview-only is already "
            "the default."
        ),
    )
    publication_mode.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Explicitly publish the validated next ordinal; without this "
            "flag, the command only previews it."
        ),
    )
    return parser


def _publication_result(
    *,
    stage_root: Path,
    context: Stage6SourceRepairContext,
) -> dict[str, object]:
    result: dict[str, object] = {
        "formal_run_id": FIXED_FORMAL_RUN_ID,
        "stage_root": str(stage_root),
        "amendment_sha256": context.amendment_sha256,
        "amendment_size_bytes": context.amendment_size_bytes,
        "cutover": {
            "seed": 20260716,
            "last_origin_update": 9,
            "first_repaired_update": 10,
        },
    }
    if context.continuation_sha256 is not None:
        result.update(
            {
                "continuation_sha256": context.continuation_sha256,
                "continuation_size_bytes": context.continuation_size_bytes,
                "repair_ordinal": 2,
                "next_resource_attempt": 3,
            }
        )
    supplement_sha256 = getattr(context, "supplement_sha256", None)
    if supplement_sha256 is not None:
        result.update(
            {
                "supplement_sha256": supplement_sha256,
                "supplement_size_bytes": getattr(
                    context,
                    "supplement_size_bytes",
                    None,
                ),
                "repair_ordinal": 3,
                "accepted_prefix_updates": 32,
                "failed_update": 33,
                "next_resource_attempt": 2,
            }
        )
    closure_sha256 = getattr(context, "closure_sha256", None)
    if closure_sha256 is not None:
        result.update(
            {
                "closure_sha256": closure_sha256,
                "closure_size_bytes": getattr(
                    context,
                    "closure_size_bytes",
                    None,
                ),
                "repair_ordinal": 4,
                "accepted_prefix_updates": 32,
                "failed_update": 33,
                "next_resource_attempt": 3,
                "next_segment_index": 5,
            }
        )
    frontier_recovery_sha256 = getattr(
        context,
        "frontier_recovery_sha256",
        None,
    )
    if frontier_recovery_sha256 is not None:
        result.update(
            {
                "frontier_recovery_sha256": frontier_recovery_sha256,
                "frontier_recovery_size_bytes": getattr(
                    context,
                    "frontier_recovery_size_bytes",
                    None,
                ),
                "repair_ordinal": 5,
                "accepted_prefix_updates": 46,
                "failed_update": 47,
                "next_resource_attempt": 2,
                "next_segment_index": 6,
            }
        )
    sensor_acceleration_sha256 = getattr(
        context,
        "sensor_acceleration_sha256",
        None,
    )
    if sensor_acceleration_sha256 is not None:
        result.update(
            {
                "sensor_acceleration_sha256": sensor_acceleration_sha256,
                "sensor_acceleration_size_bytes": getattr(
                    context,
                    "sensor_acceleration_size_bytes",
                    None,
                ),
                "coverage_cache_manifest_path": getattr(
                    context,
                    "coverage_cache_manifest_path",
                    None,
                ),
                "coverage_cache_manifest_sha256": getattr(
                    context,
                    "coverage_cache_manifest_sha256",
                    None,
                ),
                "coverage_cache_manifest_size_bytes": getattr(
                    context,
                    "coverage_cache_manifest_size_bytes",
                    None,
                ),
                "coverage_cache_entry_set_sha256": getattr(
                    context,
                    "coverage_cache_entry_set_sha256",
                    None,
                ),
                "repair_ordinal": 6,
                "accepted_prefix_updates": 49,
                "failed_update": 50,
                "next_resource_attempt": 2,
                "next_segment_index": 7,
            }
        )
    return result


def _current_immutable_bindings(
    *,
    identity: dict[str, object],
    review: dict[str, object],
    stage5_gate_sha256: str,
) -> dict[str, object]:
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
    result["stage5_gate_sha256"] = stage5_gate_sha256
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
            "coverage_cache_runtime_mode": STAGE6_COVERAGE_CACHE_RUNTIME_MODE,
            "coverage_cache_formal_audit_sha256": (
                STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
            ),
            "coverage_cache_formal_audit_size_bytes": (
                STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
            ),
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo_root.expanduser().resolve()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = repo / config_path
    config_path = config_path.resolve()
    stage_root = args.stage_root.expanduser().resolve()

    authorization = verify_stage6_review_launch_authorization(
        authorization_path=args.review_authorization,
        repo_root=repo,
        config_path=config_path,
        expected_run_id=FIXED_FORMAL_RUN_ID,
    )
    authorization.require_current("source-repair amendment entry")
    identity = stage6_execution_identity(
        repo_root=repo,
        config_path=config_path,
    )
    review = validate_stage6_verified_review_authorization(
        authorization.canonical_record(),
        execution_identity=identity,
        formal_run_id=FIXED_FORMAL_RUN_ID,
    )
    config = load_stage6_config(config_path)
    immutable = _current_immutable_bindings(
        identity=identity,
        review=review,
        stage5_gate_sha256=config.stage5_authority.gate_sha256,
    )
    context = create_stage6_source_repair_amendment(
        stage_root=stage_root,
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
        publish=args.publish,
    )
    authorization.require_current("source-repair amendment exit")
    if args.publish:
        context.require_current("source-repair amendment exit")
    print(
        json.dumps(
            _publication_result(stage_root=stage_root, context=context),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
