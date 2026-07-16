# Xunce Stage Documentation Index

This file defines where Xunce stage information belongs. Keep it short and use it as the routing map before adding new documentation.

## File Boundaries

| Location | Responsibility |
|---|---|
| `AGENTS.md` | Agent long-term operating rules, current hard boundaries, current mainline contracts, and only the most recent key stage summaries. |
| `README.md` | Human-facing project overview, current route, and documentation entry points. |
| `docs/superpowers/plans` | Full implementation plans for each stage. These are intent and execution instructions, not proof of results. |
| `outputs/.../report.md` | Real execution results for each stage run. These are the evidence source for status, metrics, blockers, and next routes. |
| `docs/算法设计与系统架构报告.md` | Architecture design, system evolution, and durable design decisions. It should not accumulate every smoke-run execution log. |
| `docs/superpowers/specs` | Long-lived technical specifications and interface contracts. It should summarize current contracts, not repeat every runner result. |
| `configs/stage_registry.json` | Machine-readable stage registration only. Do not use it as narrative documentation. |

## Path Planner v2 Gate Route

| Location | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-07-16-multiplatform-path-planner-v2-design.md` | Durable Path Planner v2 design and safety-boundary contract. |
| `docs/superpowers/plans/2026-07-16-multiplatform-path-planner-v2-implementation.md` | Task-by-task implementation intent and acceptance sequence. |
| `D:/xunce/out/path_v2/g0/report.md` | Gate 0 clean-worktree baseline and isolation result. |
| `D:/xunce/out/path_v2/g0/manifest.json` | Hash and size bindings for the Gate 0 artifact set, excluding the manifest itself. |

## Current Stage26 Mainline

```text
Stage26.0 -> Stage26.1 -> Stage26.2 -> Stage26.3 -> Stage26.4 -> Stage26.5
```

Short form:

```text
Stage26.0 -> Stage26.5
```

Current next route:

```text
repair_stage26_synthetic_exploration_credit_assignment
```

Stage26.5 diagnosed that the best synthetic terrain candidate was not sampled as a trainable selected action, so it did not receive direct PPO credit. It also found missing candidate-level synthetic LOS / hard obstacle / Hybrid A* path-cost feature exposure.

## Lookup Rules

- To understand what the project currently does, read `README.md`.
- To understand why the system is designed this way, read `docs/算法设计与系统架构报告.md` and the relevant spec under `docs/superpowers/specs`.
- To implement a specific stage, read its plan under `docs/superpowers/plans`.
- To verify what actually happened in a stage run, read `outputs/.../report.md` and the stage summary JSON.
- To know which registered stage command is executable, read `configs/stage_registry.json` or run `python scripts/run_stage.py --list`.
- To know what a future agent must not break, read `AGENTS.md`.

## Update Policy

- Do not append full stage plans to `AGENTS.md` or `README.md`.
- Do not copy full output reports into long-form docs.
- Prefer linking or naming the exact plan/report/root instead of duplicating metrics.
- Keep this index updated when adding a new documentation category or changing file responsibilities.
