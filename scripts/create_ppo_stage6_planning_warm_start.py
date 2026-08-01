"""创建独立的 Stage 6 U74 -> child U75 planning warm-start artifact。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lunar_exploration_ppo.utils.artifact_io import ArtifactStore
from lunar_exploration_ppo.workflows.stage6_planning_warm_start import (
    CHILD_BEST_VALIDATION_UPDATES,
    CHILD_FINAL_UPDATE,
    CHILD_FIRST_UPDATE,
    DISCARDED_STATE_NAMES,
    DISCARDED_U75_ATTEMPT,
    DISCARDED_U75_SEGMENT_INDEX,
    DISCARDED_U75_TRANSACTION_KEY,
    IMPORTED_STATE_NAMES,
    NEW_SEMANTICS_UPDATE_COUNT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_COMPLETE_SHA256,
    PARENT_CONFIG_SHA256,
    PARENT_LINEAGE_SHA256,
    PARENT_MANIFEST_SHA256,
    PARENT_POLICY_STATE_SHA256,
    PARENT_RECEIPT_KEY,
    PARENT_RUN_ID,
    PARENT_SEED,
    PARENT_UPDATE,
    WARM_START_MODE,
    WARM_START_SCHEMA,
    planning_effective_config_bytes,
    planning_warm_start_expected_bindings,
    validate_planning_warm_start_artifact,
    verify_parent_u74_bundle,
)
from lunar_exploration_ppo.workflows.stage6_review_authorization import (
    verify_stage6_review_launch_authorization,
)


DEFAULT_PARENT_STAGE_ROOT = Path(
    "D:/xunce/out/ppo_frontier/"
    "s6-standard-single-r1-20260718T220434Z/s6"
)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    DEFAULT_REPO_ROOT / "configs/ppo_highres_frontier_stage6_v1.json"
)


def create_planning_warm_start_artifact(
    *,
    output_path: str | Path,
    parent_stage_root: str | Path,
    child_run_id: str,
    repo_root: str | Path,
    config_path: str | Path,
    review_authorization_path: str | Path,
) -> dict[str, object]:
    """只读验证父 U74 后，排他写入 child warm-start 描述。"""

    effective_config_bytes = planning_effective_config_bytes(
        Path(config_path).read_bytes()
    )
    review_authorization = verify_stage6_review_launch_authorization(
        authorization_path=review_authorization_path,
        repo_root=repo_root,
        config_path=config_path,
        expected_run_id=child_run_id,
        effective_config_bytes=effective_config_bytes,
    )
    review_authorization.require_current(
        "planning warm-start artifact creation"
    )
    bindings = planning_warm_start_expected_bindings(
        repo_root=repo_root,
        review_authorization_handle=review_authorization,
    )
    child_effective_config_sha256 = hashlib.sha256(
        effective_config_bytes
    ).hexdigest()
    parent = verify_parent_u74_bundle(parent_stage_root)
    if (
        parent.resource_accepted is not True
        or parent.u75_attempt1_discarded is not True
    ):
        raise RuntimeError("parent U74 acceptance/discard boundary drifted")
    artifact: dict[str, object] = {
        "schema_version": WARM_START_SCHEMA,
        "mode": WARM_START_MODE,
        "parent": {
            "run_id": PARENT_RUN_ID,
            "seed": PARENT_SEED,
            "update": PARENT_UPDATE,
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "complete_sha256": PARENT_COMPLETE_SHA256,
            "policy_state_sha256": PARENT_POLICY_STATE_SHA256,
            "config_sha256": PARENT_CONFIG_SHA256,
            "lineage_sha256": PARENT_LINEAGE_SHA256,
            "resource_accepted": True,
            "receipt_transaction_key": PARENT_RECEIPT_KEY,
            "source_repair_last_ordinal": 6,
        },
        "discarded_u75_attempt1": {
            "transaction_key": DISCARDED_U75_TRANSACTION_KEY,
            "attempt": DISCARDED_U75_ATTEMPT,
            "segment_index": DISCARDED_U75_SEGMENT_INDEX,
            "only_phase": "pre",
            "accepted": False,
            "discarded": True,
        },
        "child": {
            "run_id": child_run_id,
            "seed": PARENT_SEED,
            "first_update": CHILD_FIRST_UPDATE,
            "final_update": CHILD_FINAL_UPDATE,
            "new_semantics_update_count": NEW_SEMANTICS_UPDATE_COUNT,
            "effective_config_sha256": child_effective_config_sha256,
        },
        "state_transfer": {
            "imported": list(IMPORTED_STATE_NAMES),
            "discarded": list(DISCARDED_STATE_NAMES),
            "child_vector_env_reset_count": 8,
            "child_best_validation_updates": list(
                CHILD_BEST_VALIDATION_UPDATES
            ),
        },
        "bindings": dict(bindings),
    }
    validate_planning_warm_start_artifact(
        artifact,
        expected_child_run_id=child_run_id,
        expected_child_config_bytes=effective_config_bytes,
        expected_bindings=bindings,
    )
    review_authorization.require_current(
        "planning warm-start artifact before write"
    )
    payload = (
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output = Path(output_path).expanduser().resolve()
    published = ArtifactStore(output.parent).write_bytes_exclusive(
        output.name,
        payload,
    )
    if published != output or published.read_bytes() != payload:
        raise RuntimeError("planning warm-start artifact write verification failed")
    review_authorization.require_current(
        "planning warm-start artifact after write"
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="创建独立 Stage 6 planning-safety child warm-start artifact。"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--parent-stage-root",
        type=Path,
        default=DEFAULT_PARENT_STAGE_ROOT,
    )
    parser.add_argument("--child-run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--review-authorization", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    create_planning_warm_start_artifact(
        output_path=args.output,
        parent_stage_root=args.parent_stage_root,
        child_run_id=args.child_run_id,
        repo_root=args.repo_root,
        config_path=args.config,
        review_authorization_path=args.review_authorization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
