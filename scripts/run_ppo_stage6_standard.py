"""启动 Stage 6 Standard v1 正式训练与评估工作流。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar_exploration_ppo.workflows.stage6 import (
    CANONICAL_STAGE5_GATE,
    STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256,
    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_PATH,
    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256,
    STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES,
    STAGE6_COVERAGE_CACHE_MANIFEST_PATH,
    STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
    STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES,
    STAGE6_COVERAGE_CACHE_ROOT,
    STAGE6_COVERAGE_CACHE_RUNTIME_MODE,
    Stage6WorkflowError,
    run_stage6_workflow,
    validate_stage6_formal_run_id,
)


def _formal_run_id(value: str) -> str:
    try:
        return validate_stage6_formal_run_id(value)
    except Stage6WorkflowError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 Stage 6 Standard v1 正式训练与评估工作流。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ppo_highres_frontier_stage6_v1.json"),
    )
    parser.add_argument("--run-id", required=True, type=_formal_run_id)
    parser.add_argument(
        "--review-authorization",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--planning-warm-start",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--planning-child-source-repair",
        type=Path,
    )
    parser.add_argument(
        "--planning-child-source-repair-continuation",
        type=Path,
    )
    parser.add_argument(
        "--stage5-gate",
        type=Path,
        default=CANONICAL_STAGE5_GATE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_stage6_workflow(
        config_path=args.config,
        run_id=args.run_id,
        stage5_gate_path=args.stage5_gate,
        review_authorization_path=args.review_authorization,
        planning_warm_start_path=args.planning_warm_start,
        planning_child_source_repair_path=(
            args.planning_child_source_repair
        ),
        planning_child_source_repair_continuation_path=(
            args.planning_child_source_repair_continuation
        ),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "stage_root": str(result.stage_root),
                "summary": result.summary,
                "routing": result.routing,
                "coverage_cache": {
                    "runtime_mode": STAGE6_COVERAGE_CACHE_RUNTIME_MODE,
                    "manifest_path": STAGE6_COVERAGE_CACHE_MANIFEST_PATH.as_posix(),
                    "manifest_sha256": STAGE6_COVERAGE_CACHE_MANIFEST_SHA256,
                    "manifest_size_bytes": (
                        STAGE6_COVERAGE_CACHE_MANIFEST_SIZE_BYTES
                    ),
                    "cache_root": STAGE6_COVERAGE_CACHE_ROOT.as_posix(),
                    "entry_set_sha256": STAGE6_COVERAGE_CACHE_ENTRY_SET_SHA256,
                    "formal_audit_path": (
                        STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_PATH.as_posix()
                    ),
                    "formal_audit_sha256": (
                        STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SHA256
                    ),
                    "formal_audit_size_bytes": (
                        STAGE6_COVERAGE_CACHE_FORMAL_AUDIT_SIZE_BYTES
                    ),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
