# Visual Workbench Design

## Summary

`visual-workbench` is the fourth, visually focused subproject for
`lunar-path-planning`. It is a local sibling directory for now, not a Git
submodule, because no remote is configured. Its role is to make existing
artifact evidence browsable, explainable, and presentable without entering the
algorithmic loop owned by `dev-platform-constraints`, `model-explorer`, and
`path-planner`.

The first version uses a React + TypeScript + Vite frontend and a FastAPI +
Pydantic backend. It opens directly as a workbench, not as a landing page, and
supports both dense research mode and lower-noise presentation mode.

## Interfaces

The backend scans only allowlisted roots: the parent `outputs/`, each existing
subproject `outputs/`, and operator-configured paths supplied through
`VISUAL_WORKBENCH_ARTIFACT_ROOTS`. It detects `schema_version` for:

- `model-explorer-contract/v1`
- `path-planner-sidecar/v1`
- `path-planner-route/v1`
- `path-feedback-manifest/v1`
- `path-feedback-summary/v1`
- `model-explorer-experiment/v1`

Initial API surface:

- `GET /api/health`
- `GET /api/project/status`
- `GET /api/artifacts`
- `GET /api/artifacts/{artifact_id}`
- `GET /api/artifacts/{artifact_id}/raw`
- `POST /api/commands/dry-run`
- `POST /api/commands/validate`

Command execution is intentionally narrow. The UI can request `dry-run` or
`validate` for `model-explorer` path-feedback and experiment manifests. Full
`run` is rejected before subprocess execution.

## UI Behavior

- Overview: subproject presence, artifact counts, schema counts, latest evidence.
- Evidence Browser: searchable artifact table with schema, status, type, and raw path in research mode.
- Map & Route Explorer: loads sidecar/route raw JSON on demand and renders cost, passability, raw path, smoothed path, and optimized path overlays on Canvas.
- Path Feedback Lab: shows path-feedback manifests/summaries and executes only white-listed `dry-run`/`validate`.
- Experiment Matrix: lists `model-explorer-experiment/v1` artifacts and presents matrix readiness without running experiments.
- Presentation Mode: reduces raw-path noise while preserving metric semantics.

## Verification

Required checks:

```powershell
cd visual-workbench
python -m pytest
cd web
npm test
npm run build
```

The backend tests cover schema indexing, raw reads, path allowlisting,
project-status reporting, dry-run/validate command construction, manifest
allowlist enforcement, and full-run rejection. Frontend tests cover workbench
startup, evidence rows, filtering, and presentation-mode switching.

## Boundaries

This subproject is read-first and artifact-first. It must not start PPO,
training, staged release, or full experiment runs. It must not change network,
action space, default A*, or existing subproject responsibilities. It must not
claim Ackermann-feasible trajectories, and it must not treat IRIS/GCS or
path-planner diagnostics as training-release evidence or performance proof.
