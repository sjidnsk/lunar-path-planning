"""Thin CLI for the Stage 4 rollout/PPO machine acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar_exploration_ppo.workflows.stage4 import run_stage4_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ppo_highres_frontier_stage4_v1.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage3-gate", required=True)
    parser.add_argument("--output-root")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = run_stage4_workflow(
        config_path=Path(arguments.config),
        run_id=arguments.run_id,
        stage3_gate_path=Path(arguments.stage3_gate),
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
                "route": "awaiting_independent_review",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
