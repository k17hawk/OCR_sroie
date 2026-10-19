"""FastAPI router for /projects/*. File-backed; no SQL session."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.api import (
    DocumentHeader,
    FileResultRead,
    ImportResponse,
    ProjectCreate,
    ProjectRead,
    ProjectUpdate,
    TypeListResponse,
    UploadResponse,
)
from app.services import documents as doc_service
from app.services import export as export_service
from app.services import import_ as import_service
from app.services import ingest as ingest_service
from app.services import projects as project_service
from app.storage import get_storage

router = APIRouter(prefix="/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.post("", response_model=ProjectRead, status_code=201)
def create_project(body: ProjectCreate) -> project_service.Project:
    if project_service.get_by_name(body.name):
        raise HTTPException(409, "project name already exists")
    kwargs = body.model_dump(exclude_none=True, by_alias=False)
    kwargs.pop("name", None)
    try:
        return project_service.create(body.name, **kwargs)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("", response_model=list[ProjectRead])
def list_projects() -> list[project_service.Project]:
    return project_service.list_all()


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: int) -> project_service.Project:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    return p


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(project_id: int, body: ProjectUpdate) -> project_service.Project:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")

    patch = body.model_dump(exclude_unset=True, by_alias=False)
    if not patch:
        return p

    new_name = patch.get("name")
    if new_name and new_name != p.name:
        existing = project_service.get_by_name(new_name)
        if existing and existing.id != p.id:
            raise HTTPException(409, "project name already exists")

    return project_service.update(p, patch)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@router.get("/{project_id}/types", response_model=TypeListResponse)
def get_project_types(project_id: int) -> TypeListResponse:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    return TypeListResponse(
        doc_types=p.doc_types,
        field_types=p.field_types,
        layout_types=p.layout_types,
    )


@router.put("/{project_id}/types", response_model=TypeListResponse)
def set_project_types(project_id: int, body: TypeListResponse) -> TypeListResponse:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    p.doc_types = list(body.doc_types)
    p.field_types = list(body.field_types)
    p.layout_types = list(body.layout_types)
    project_service.update(p, {
        "doc_types": p.doc_types,
        "field_types": p.field_types,
        "layout_types": p.layout_types,
    })
    return TypeListResponse(
        doc_types=p.doc_types,
        field_types=p.field_types,
        layout_types=p.layout_types,
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@router.get("/{project_id}/documents", response_model=list[DocumentHeader])
def list_documents(
    project_id: int,
    status: str | None = Query(None),
) -> list[DocumentHeader]:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    storage = get_storage()
    out: list[DocumentHeader] = []
    for row in doc_service.list_for_project(project_id):
        if status and row.status != status:
            continue
        out.append(DocumentHeader(
            document_id=row.document_id,
            page_number=row.page_number,
            page_count=row.page_count,
            revision=row.revision,
            status=row.status,
            image_url=storage.url(row.image_key),
            source_file_id=row.source_file_id,
            source_filename=row.source_filename,
            updated_at=row.updated_at,
        ))
    return out


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/{project_id}/images", response_model=UploadResponse)
async def upload_images(
    project_id: int,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")

    results: list[FileResultRead] = []
    for f in files:
        data = await f.read()
        file_results = ingest_service.ingest_upload(
            p, filename=f.filename or "upload", data=data
        )
        for r in file_results:
            results.append(FileResultRead(
                filename=r.filename,
                ok=r.ok,
                error=r.error,
                skipped_duplicate=r.skipped_duplicate,
                warning=r.warning,
                document_ids=r.document_ids,
            ))

    storage = get_storage()
    documents: list[DocumentHeader] = []
    for r in results:
        for doc_id in r.document_ids:
            row = doc_service.get(doc_id)
            if row is None:
                continue
            documents.append(DocumentHeader(
                document_id=row.document_id,
                page_number=row.page_number,
                page_count=row.page_count,
                revision=row.revision,
                status=row.status,
                image_url=storage.url(row.image_key),
                source_file_id=row.source_file_id,
                source_filename=row.source_filename,
                updated_at=row.updated_at,
            ))

    return UploadResponse(results=results, documents=documents)


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------

@router.get("/{project_id}/export")
def export_project(
    project_id: int,
    format: str = Query("jsonl", pattern="^(jsonl|zip)$"),
    status: list[str] | None = Query(None),
    include_images: bool = Query(False),
) -> StreamingResponse:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")

    if format == "zip" or include_images:
        body = export_service.stream_zip_with_images(project_id, statuses=status)
        return StreamingResponse(
            body, media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="project_{project_id}.zip"'},
        )
    body = export_service.stream_jsonl(project_id, statuses=status)
    return StreamingResponse(
        body, media_type="application/x-ndjson",
        headers={"Content-Disposition":
                 f'attachment; filename="project_{project_id}.jsonl"'},
    )


@router.post("/{project_id}/import", response_model=ImportResponse)
async def import_project(
    project_id: int,
    file: UploadFile = File(...),
    overwrite: bool = Query(False),
) -> ImportResponse:
    p = project_service.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    result = import_service.import_jsonl(p, file.file, overwrite=overwrite)
    return ImportResponse(**result)