from __future__ import annotations

import argparse

from lunar_exploration_ppo.workflows.foundation import run_foundation_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Foundation machine preflight only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-output-root")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    result = run_foundation_preflight(
        config_path=args.config,
        base_output_root=args.base_output_root,
        run_id=args.run_id,
    )
    print(f"Foundation preflight output: {result.stage_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
