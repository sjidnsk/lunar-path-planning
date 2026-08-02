"""Thin CLI for the Stage 5 fair-baseline machine acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar_exploration_ppo.workflows.stage5 import run_stage5_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ppo_highres_frontier_stage5_v1.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage4-gate", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = run_stage5_workflow(
        config_path=Path(arguments.config),
        run_id=arguments.run_id,
        stage4_gate_path=Path(arguments.stage4_gate),
        checkpoint_root=Path(arguments.checkpoint_root),
        base_output_root=(
            Path(arguments.output_root) if arguments.output_root is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "stage_root": str(result.stage_root),
                "state": result.summary["state"],
                "route": result.routing["route"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
