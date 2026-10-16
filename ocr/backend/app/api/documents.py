"""FastAPI router for /documents/*. File-backed; no SQL session."""
from __future__ import annotations

import copy
from io import BytesIO

from fastapi import APIRouter, HTTPException
import numpy as np
from PIL import Image as PILImage

from app.core.validate import validate
from app.pipeline import registry
from app.pipeline.table_ocr import recognize_table
from app.schemas.api import (
    AnnotationPatch,
    AnnotationPut,
    DocumentRead,
    OcrRegionRequest,
    OcrRegionResponse,
    PageFixRequest,
    PageFixResponse,
    StatusChange,
    TableOcrRequest,
    TableOcrResponse,
    ValidateResponse,
)
from app.schemas.doc_v2 import Doc
from app.services import annotations as ann_service
from app.services import documents as doc_service
from app.services.lines import TEXT_TYPES, detect_lines
from app.services import projects as project_service
from app.services import status as status_service
from app.storage import get_storage

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(document_id: str) -> doc_service.doc_store.DocumentRow:
    row = doc_service.get(document_id)
    if row is None:
        raise HTTPException(404, "document not found")
    return row


def _read_doc_response(row: doc_service.doc_store.DocumentRow) -> DocumentRead:
    return DocumentRead(
        document_id=row.document_id,
        revision=row.revision,
        status=row.status,
        image_url=get_storage().url(row.image_key),
        source_file_id=row.source_file_id,
        doc=row.doc,
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: str) -> DocumentRead:
    return _read_doc_response(_load(document_id))


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

@router.put("/{document_id}/annotation", response_model=DocumentRead)
def put_annotation(document_id: str, body: AnnotationPut) -> DocumentRead:
    row = _load(document_id)
    project = project_service.get(row.project_id)
    if project is None:
        raise HTTPException(500, "project missing for document")
    row = ann_service.replace(
        row=row, project=project,
        payload=body.doc,
        revision=body.revision, user_id=None,
    )
    return _read_doc_response(row)


@router.patch("/{document_id}/annotation", response_model=DocumentRead)
def patch_annotation(document_id: str, body: AnnotationPatch) -> DocumentRead:
    row = _load(document_id)
    project = project_service.get(row.project_id)
    if project is None:
        raise HTTPException(500, "project missing for document")
    row = ann_service.patch(
        row=row, project=project, ops=body.ops,
        revision=body.revision, user_id=None,
    )
    return _read_doc_response(row)


@router.post("/{document_id}/validate", response_model=ValidateResponse)
def validate_document(document_id: str) -> ValidateResponse:
    row = _load(document_id)
    issues = validate(row.doc)
    return ValidateResponse(
        ok=not any(i.level == "error" for i in issues), issues=issues
    )


@router.post("/{document_id}/status", response_model=DocumentRead)
def change_status(document_id: str, body: StatusChange) -> DocumentRead:
    row = _load(document_id)
    row = status_service.change(
        row=row, target=body.target, user_id=None, comment=body.comment,
    )
    return _read_doc_response(row)


# ---------------------------------------------------------------------------
# Pre-annotation
# ---------------------------------------------------------------------------

@router.post("/{document_id}/preannotate", status_code=202)
def preannotate(document_id: str, body: dict):
    from app.jobs import runner as job_runner
    from app.schemas.api import JobRead

    row = _load(document_id)
    model_settings = body.get("model_settings") or {"family": "mock"}
    device = body.get("device", "cpu")

    job = job_runner.enqueue_preannotate(
        document_id=row.document_id,
        family=model_settings.get("family", "mock"),
        device=device,
    )
    return JobRead(
        id=job["id"],
        document_id=job["document_id"],
        kind=job["kind"],
        status=job["status"],
        progress=job["progress"],
        device=job["device"],
    )


# ---------------------------------------------------------------------------
# Region OCR
# ---------------------------------------------------------------------------

@router.post("/{document_id}/ocr-region", response_model=OcrRegionResponse)
def ocr_region(document_id: str, body: OcrRegionRequest) -> OcrRegionResponse:
    row = _load(document_id)
    image_bytes = get_storage().get(row.image_key)
    img = PILImage.open(BytesIO(image_bytes))
    img.load()

    xs = [p[0] for p in body.quad]
    ys = [p[1] for p in body.quad]
    x0, y0, x1, y1 = max(0, int(min(xs))), max(0, int(min(ys))), int(max(xs)), int(max(ys))
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(422, "empty quad")
    crop = img.crop((x0, y0, x1, y1))

    try:
        adapters = registry.load({"family": body.family, "device": body.device})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    try:
        results = adapters["rec"].recognize([crop], lang_hint=body.lang_hint)
    except RuntimeError as e:
        raise HTTPException(501, str(e)) from e

    text, conf = results[0] if results else ("", 0.0)
    return OcrRegionResponse(
        text=text, confidence=float(conf),
        family=body.family, device=body.device,
        model=getattr(adapters["rec"], "name", body.family),
    )


# ---------------------------------------------------------------------------
# Table OCR
# ---------------------------------------------------------------------------

@router.post("/{document_id}/ocr-table", response_model=TableOcrResponse)
def ocr_table(document_id: str, body: TableOcrRequest) -> TableOcrResponse:
    row = _load(document_id)
    image_bytes = get_storage().get(row.image_key)
    img = PILImage.open(BytesIO(image_bytes))
    img.load()

    try:
        adapters = registry.load({"family": body.family, "device": body.device})
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    try:
        table = recognize_table(
            doc=row.doc, img=img, quad=body.quad, adapter=adapters["table"],
            rec=adapters.get("rec"), lang_hint=body.lang_hint, html=body.html,
        )
    except RuntimeError as e:
        raise HTTPException(501, str(e)) from e

    return TableOcrResponse(
        table=table, family=body.family, device=body.device,
        model=getattr(adapters["table"], "name", body.family),
    )

# ---------------------------------------------------------------------------
# Geometry fixes that read the scan: detect lines, snap table
# ---------------------------------------------------------------------------

def _gray(row: doc_service.doc_store.DocumentRow) -> np.ndarray:
    img = PILImage.open(BytesIO(get_storage().get(row.image_key)))
    img.load()
    gray = np.asarray(img.convert("L"))
    if (gray.shape[1], gray.shape[0]) != (row.doc.width, row.doc.height):
        raise HTTPException(
            409, f"image is {gray.shape[1]}x{gray.shape[0]} but the page says "
                 f"{row.doc.width}x{row.doc.height}",
        )
    return gray


def _page_record(row: doc_service.doc_store.DocumentRow, body: PageFixRequest) -> dict:
    return copy.deepcopy(body.doc) if body.doc is not None else row.doc.model_dump(mode="json")


def _region(rec: dict, element_id: int) -> dict:
    for el in rec.get("layout") or []:
        if isinstance(el, dict) and el.get("id") == element_id:
            return el
    raise HTTPException(404, f"region {element_id} not found")


@router.post("/{document_id}/detect-lines", response_model=PageFixResponse)
def detect_lines_endpoint(document_id: str, body: PageFixRequest) -> PageFixResponse:
    """One ``lines[]`` record per printed line inside each text region."""
    row = _load(document_id)
    rec = _page_record(row, body)
    ids = None
    warnings: list[str] = []
    if body.element_id is not None:
        el = _region(rec, body.element_id)
        if el.get("type") not in TEXT_TYPES:
            warnings.append(f"block {body.element_id}: a {el.get('type')} has no text lines to detect")
        ids = {body.element_id}
    rec, more = detect_lines(_gray(row), rec, ids)
    return PageFixResponse(record=rec, warnings=warnings + more)
