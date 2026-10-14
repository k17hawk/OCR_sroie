"""Request and response bodies for the REST API.

Kept separate from the doc schema (``doc_v2.py``) so external clients can
generate against two stable OpenAPI specs: the annotation schema and the
API surface.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.doc_v2 import Doc, Quad, Table


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    doc_types: list[str] | None = None
    field_types: list[str] | None = None
    layout_types: list[str] | None = None
    label_map: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None


class ProjectUpdate(BaseModel):
    """Partial update of a project. Every field is optional; only the keys
    present in the request body are applied."""
    name: str | None = None
    doc_types: list[str] | None = None
    field_types: list[str] | None = None
    layout_types: list[str] | None = None
    label_map: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None
    max_long_edge: int | None = None
    default_dpi: int | None = None
    default_language: str | None = None
    max_upload_mb: int | None = None
    require_doc_type_on_submit: bool | None = None


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    doc_types: list[str]
    field_types: list[str]
    layout_types: list[str]
    label_map: dict[str, Any]
    model_settings: dict[str, Any]
    max_long_edge: int | None = None
    default_dpi: int = 200
    default_language: str = "ne"
    max_upload_mb: int = 100
    require_doc_type_on_submit: bool = True
    created_at: datetime
    updated_at: datetime


class TypeListResponse(BaseModel):
    """Compact response for the `/projects/{id}/types` endpoint."""
    doc_types: list[str]
    field_types: list[str]
    layout_types: list[str]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentHeader(BaseModel):
    """Lightweight listing entry (no full doc)."""
    document_id: str
    page_number: int
    page_count: int
    revision: int
    status: str
    image_url: str
    source_file_id: str | None = None


class DocumentRead(BaseModel):
    document_id: str
    revision: int
    status: str
    image_url: str
    source_file_id: str | None = None
    doc: Doc


# ---------------------------------------------------------------------------
# Annotation PUT / PATCH
# ---------------------------------------------------------------------------

class AnnotationPut(BaseModel):
    revision: int
    # Raw editor JSON on purpose. Typing this as ``Doc`` made FastAPI reject
    # the save (422) BEFORE ``annotations._clean_payload`` could fix it - a
    # dragged box with a .5 coordinate or one leftover editor key was enough,
    # and the toolbar showed nothing. The service cleans, then validates.
    doc: dict[str, Any]


class AnnotationPatch(BaseModel):
    revision: int
    ops: list[dict[str, Any]]


class StatusChange(BaseModel):
    target: str
    comment: str | None = None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidateIssue(BaseModel):
    level: str  # "error" | "warning"
    path: str
    message: str


class ValidateResponse(BaseModel):
    ok: bool
    issues: list[ValidateIssue]


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class FileResultRead(BaseModel):
    filename: str
    ok: bool
    error: str | None = None
    skipped_duplicate: bool = False
    document_ids: list[str] = []


class UploadResponse(BaseModel):
    results: list[FileResultRead]
    documents: list[DocumentHeader]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobRead(BaseModel):
    id: int
    document_id: str
    kind: str
    status: str
    progress: float
    device: str
    result: dict[str, Any] | None = None
    error: str | None = None


class PreannotateRequest(BaseModel):
    model_settings: dict[str, Any] = Field(default_factory=dict)
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class ImportResponse(BaseModel):
    imported: int
    skipped: int
    errors: int


# ---------------------------------------------------------------------------
# Region / table OCR
# ---------------------------------------------------------------------------

class OcrRegionRequest(BaseModel):
    quad: Quad
    family: str = "mock"
    device: str = "cpu"
    lang_hint: str = "ne"


class OcrRegionResponse(BaseModel):
    text: str
    confidence: float
    family: str
    device: str
    model: str


class TableOcrRequest(BaseModel):
    quad: Quad
    family: str = "mock"
    device: str = "cpu"
    lang_hint: str = "ne"
    # Table HTML typed by the annotator: its rows / columns / spans / text
    # are kept and only placed on the image (ruling lines, text rows).
    html: str | None = None


class TableOcrResponse(BaseModel):
    table: Table
    family: str
    device: str
    model: str