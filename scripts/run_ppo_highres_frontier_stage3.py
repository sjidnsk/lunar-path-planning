"""Run the bounded Stage 3 CUDA machine audit."""

from __future__ import annotations

import argparse
import json

from lunar_exploration_ppo.workflows.stage3 import run_stage3_workflow


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 3 forward/action-distribution machine checks."
    )
    parser.add_argument(
        "--config",
        default="configs/ppo_highres_frontier_stage3_v1.json",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage2-gate",
        required=True,
        help="Path to the externally persisted verified Stage 2 authority artifact.",
    )
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args()
    result = run_stage3_workflow(
        config_path=arguments.config,
        run_id=arguments.run_id,
        stage2_gate_path=arguments.stage2_gate,
        base_output_root=arguments.output_root,
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
