from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from .artifacts import ArtifactService, WorkbenchSettings
from .commands import CommandRejected, CommandRequest, CommandService


def create_app(settings: WorkbenchSettings | None = None) -> FastAPI:
    settings = settings or WorkbenchSettings()
    artifacts = ArtifactService(settings)
    commands = CommandService(settings)
    app = FastAPI(title="Visual Workbench API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/project/status")
    def project_status() -> dict:
        repo_root = settings.normalized_repo_root()
        subprojects = {
            name: {
                "exists": (repo_root / name).exists(),
                "path": str(repo_root / name),
            }
            for name in ("dev-platform-constraints", "model-explorer", "path-planner")
        }
        return {
            "repo_root": str(repo_root),
            "subprojects": subprojects,
            "artifact_roots": [str(root) for root in settings.resolved_artifact_roots()],
            "submodule_note": "visual-workbench is local until a fourth-submodule remote is configured.",
        }

    @app.get("/api/artifacts")
    def list_artifacts() -> dict:
        return {"artifacts": [artifact.model_dump() for artifact in artifacts.list_artifacts()]}

    @app.get("/api/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict:
        try:
            return artifacts.get_artifact(artifact_id).model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/api/artifacts/{artifact_id}/raw")
    def get_raw(artifact_id: str) -> Response:
        try:
            raw = artifacts.get_raw(artifact_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(content=raw.content, media_type=raw.media_type)

    @app.post("/api/commands/dry-run")
    def dry_run(request: CommandRequest) -> dict:
        return _execute(commands, request, forced_action="dry-run")

    @app.post("/api/commands/validate")
    def validate(request: CommandRequest) -> dict:
        return _execute(commands, request, forced_action="validate")

    return app


def _execute(commands: CommandService, request: CommandRequest, *, forced_action: str) -> dict:
    if request.action != forced_action:
        raise HTTPException(status_code=400, detail=f"endpoint only accepts action={forced_action}; got {request.action}")
    try:
        return commands.execute(request).model_dump()
    except (CommandRejected, PermissionError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("visual_workbench.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
