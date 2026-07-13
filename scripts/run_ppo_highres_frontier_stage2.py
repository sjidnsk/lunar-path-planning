"""Run the bounded Stage 2 machine workflow; authority remains a separate action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar_exploration_ppo.workflows.stage2 import run_stage2_workflow, verify_stage2_machine_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PPO high-resolution frontier Stage 2 machine workflow.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    result = run_stage2_workflow(
        config_path=arguments.config,
        run_id=arguments.run_id,
        base_output_root=arguments.output_root,
    )
    verify_stage2_machine_run(stage_root=result.stage_root, repo_root=arguments.repo_root)
    print(json.dumps({"run_id": result.run_id, "stage_root": str(result.stage_root), "route": "awaiting_independent_review"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
