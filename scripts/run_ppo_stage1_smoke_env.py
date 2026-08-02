from __future__ import annotations

import argparse
from pathlib import Path

from lunar_exploration_ppo.workflows.stage1 import run_stage1_smoke_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage 1 Smoke v1 environment workflow")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-output-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_stage1_smoke_workflow(
        config_path=args.config,
        run_id=args.run_id,
        base_output_root=args.base_output_root,
    )
    print(result.stage_root.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
