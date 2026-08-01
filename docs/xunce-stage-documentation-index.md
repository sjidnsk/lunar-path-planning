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
| `docs/route-retirement-manifest.md` | The sole detailed entry for retirement scope and manual-cleanup boundaries. It does not authorize deletion or replace a stage report or runtime evidence. |

## Current Retained Mainline

| Route | Authoritative documentation |
|---|---|
| High-resolution frontier PPO / Stage6 | `docs/ppo-highres-frontier-stage6.md`; `docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md` |
| Midterm reduced-scale dual-gate G1/G2/G3 | `docs/xunce-midterm-dual-gate-runbook.md`; `docs/superpowers/specs/2026-07-26-midterm-dual-gate-experiment-design.md` |
| Multi-platform path planning v3 | `docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-design.md` and the v3 route below; v3 is opt-in. |
| Platform constraints | `dev-platform-constraints/`; the aligned hard slope threshold remains `max_traversable_slope_deg=30.0`. |
| Default path planning | Retained `path-planner/` Python grid A*; the PPO adapter uses `path_planner.search.AStarPlanner`. Hybrid A* and v3 are opt-in, not default runtime paths. |

## Path Planner v2 Gate Route

| Location | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-07-16-multiplatform-path-planner-v2-design.md` | Durable Path Planner v2 design and safety-boundary contract. |
| `docs/superpowers/plans/2026-07-16-multiplatform-path-planner-v2-implementation.md` | Task-by-task implementation intent and acceptance sequence. |
| `D:/xunce/out/path_v2/g0/report.md` | Gate 0 clean-worktree baseline and isolation result. |
| `D:/xunce/out/path_v2/g0/manifest.json` | Hash and size bindings for the Gate 0 artifact set, excluding the manifest itself. |

## Path Planner v3 Clean-Room Route

| Location | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-design.md` | Frozen shared-layer and three-platform algorithm, safety, fallback, commitment, and performance contract. |
| `docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-interface-schema.md` | Human-readable C++ domain model, wire mapping, identity, time, bundle, response, and benchmark contract. |
| `path-planner/schemas/v3/` | Draft 2020-12 machine-readable request, response, reference, capability, algorithm, and benchmark schemas. |
| `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-roadmap.md` | Master execution order, dependency gates, and final acceptance handoff. |
| `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-contracts-core.md` | C++20 build, contracts, immutable map, shared search, corridor, learned-cost snapshot, cache, and bounded-QP plan. |
| `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-wheel.md` | Wheeled platform plan for forward, reverse, in-place spin, smoothing, timing, and primitive fallback. |
| `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-legged.md` | Legged body-reference plan for pose-height search, product-space corridor, timing, and bounded feasibility claims. |
| `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-hopper.md` | Hopper plan for deterministic landing regions, one executable ballistic boundary, flight-tube and attitude certification. |
| `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-integration-performance.md` | Unified planner, atomic bundle, outcome/directive arbitration, codecs, system tests, and fixed-profile latency plan. |

## Retired Xunce History

Xunce Stage18--26, path-feedback, and early policy experiments are retired history and no longer accept new development entry. They remain available only for historical reproducibility and as manual-cleanup candidates; retirement does not claim that source has been deleted. For the exclusive detailed retirement scope and manual-cleanup boundary, read `docs/route-retirement-manifest.md`.

`model-explorer` and `visual-workbench` are likewise retirement candidates, while `path-planner` and `dev-platform-constraints` remain retained submodules. Their physical removal, if any, requires separate human confirmation.

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
