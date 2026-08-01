"""CLI for the Stage 6 exact coverable-mask cache prewarm workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lunar_exploration_ppo.workflows.stage6_coverage_cache import prewarm_stage6_coverage_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("D:/xunce/cache/ppo_frontier"))
    parser.add_argument("--formal-run-id", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run-scenarios", type=int)
    args = parser.parse_args(argv)
    summary = prewarm_stage6_coverage_cache(
        output_root=args.output_root, cache_root=args.cache_root, formal_run_id=args.formal_run_id,
        workers=args.workers, dry_run_scenarios=args.dry_run_scenarios,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
