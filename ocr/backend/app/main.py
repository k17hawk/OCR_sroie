from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response


from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from app import __version__
from app.api import documents, projects
from app.storage import get_storage
from app.storage.local import LocalStorage


from app.api import documents, projects, jobs
from app.api import documents, jobs, projects, system

app = FastAPI(title="DevaLabel", version=__version__)

app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(jobs.router)


app.include_router(system.router)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/files/{key:path}")
def serve_file(key: str) -> Response:
    """Local-storage file server. In S3 mode, ``image_url`` is a presigned URL."""
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        return Response(status_code=404)
    try:
        return Response(content=storage.get(key), media_type="image/jpeg")
    except FileNotFoundError:
        return Response(status_code=404)

@app.exception_handler(OperationalError)
async def sqlalchemy_operational_error_handler(
    _request: Request, exc: OperationalError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": "database not ready",
            "hint": "run `python -m scripts.seed` from backend/, or "
                    "`alembic upgrade head` if you are using migrations",
            "error": str(exc.orig),
        },
    )
