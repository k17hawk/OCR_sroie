"""Upload orchestration: validate, dedupe, store, create skeleton documents."""
from __future__ import annotations

import io
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.models.project import DEFAULT_LAYOUT_TYPES  # not used; kept for reference
from app.services import images as img_service
from app.services import pdfs as pdf_service
from app.services import quality_metrics
from app.services.projects import Project
from app.services.skeleton import make_skeleton
from app.storage import doc_store


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ALLOWED_PDF_EXTS = {".pdf"}
ALLOWED_ZIP_EXTS = {".zip"}


@dataclass
class FileResult:
    filename: str
    ok: bool
    document_ids: list[str] = field(default_factory=list)
    error: str | None = None
    skipped_duplicate: bool = False


def ingest_upload(project: Project, *, filename: str, data: bytes) -> list[FileResult]:
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_ZIP_EXTS:
        return _ingest_zip(project, filename, data)
    if ext in ALLOWED_PDF_EXTS:
        return [_ingest_pdf(project, filename, data)]
    return [_ingest_image(project, filename, data)]


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def _ingest_image(project: Project, filename: str, data: bytes) -> FileResult:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return FileResult(filename=filename, ok=False, error=f"unsupported extension {ext!r}")

    if len(data) > project.max_upload_mb * 1024 * 1024:
        return FileResult(filename=filename, ok=False,
                          error=f"file exceeds {project.max_upload_mb} MB")

    try:
        prepared = img_service.prepare_image(
            data=data, source_ext=ext, max_long_edge=project.max_long_edge
        )
    except img_service.ImageError as e:
        return FileResult(filename=filename, ok=False, error=str(e))

    existing = doc_store.find_by_sha256(project.id, prepared.sha256)
    if existing is not None:
        return FileResult(
            filename=filename, ok=True, skipped_duplicate=True,
            document_ids=[existing.document_id],
        )

    n = doc_store.next_document_number()
    image_key = img_service.store_image(n, project.id, prepared)

    quality = quality_metrics.compute_quality(prepared.image, prepared.original_bytes)
    doc = make_skeleton(
        n=n, image_ext=prepared.save_ext,
        width=prepared.width, height=prepared.height,
        page_number=1, page_count=1,
        language=project.default_language,
        quality=quality,
    )

    row = doc_store.DocumentRow(
        document_id=doc.document_id,
        project_id=project.id,
        doc_number=n,
        revision=1,
        status=doc.annotation.status,
        doc=doc,
        image_key=image_key,
        sha256=prepared.sha256,
        source_filename=filename,
    )
    doc_store.save(row)
    return FileResult(filename=filename, ok=True, document_ids=[row.document_id])


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _ingest_pdf(project: Project, filename: str, data: bytes) -> FileResult:
    if len(data) > project.max_upload_mb * 1024 * 1024:
        return FileResult(filename=filename, ok=False,
                          error=f"file exceeds {project.max_upload_mb} MB")

    try:
        pages = list(pdf_service.iter_pdf_pages(data, dpi=project.default_dpi))
    except Exception as e:
        return FileResult(filename=filename, ok=False, error=f"pdf: {e}")

    source_file_id = f"src_{uuid.uuid4().hex[:12]}"
    page_count = len(pages)
    document_ids: list[str] = []

    for page_number, jpeg_bytes, text_layer, legacy in pages:
        try:
            prepared = img_service.prepare_image(
                data=jpeg_bytes, source_ext=".jpg",
                max_long_edge=project.max_long_edge,
            )
        except img_service.ImageError as e:
            return FileResult(filename=filename, ok=False,
                              error=f"page {page_number}: {e}")

        n = doc_store.next_document_number()
        image_key = img_service.store_image(n, project.id, prepared)
        quality = quality_metrics.compute_quality(prepared.image, jpeg_bytes)
        doc = make_skeleton(
            n=n, image_ext=prepared.save_ext,
            width=prepared.width, height=prepared.height,
            page_number=page_number, page_count=page_count,
            language=project.default_language,
            quality=quality,
        )

        row = doc_store.DocumentRow(
            document_id=doc.document_id,
            project_id=project.id,
            doc_number=n,
            revision=1,
            status=doc.annotation.status,
            doc=doc,
            image_key=image_key,
            sha256=prepared.sha256,
            source_file_id=source_file_id,
            source_filename=filename,
            text_layer=text_layer or None,
            legacy_font=legacy,
        )
        doc_store.save(row)
        document_ids.append(row.document_id)

    return FileResult(filename=filename, ok=True, document_ids=document_ids)


# ---------------------------------------------------------------------------
# ZIP
# ---------------------------------------------------------------------------

def _ingest_zip(project: Project, filename: str, data: bytes) -> list[FileResult]:
    results: list[FileResult] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if Path(info.filename).suffix.lower() == ".zip":
                    results.append(FileResult(
                        filename=f"{filename}/{info.filename}",
                        ok=False, error="nested zips are not supported",
                    ))
                    continue
                results.extend(ingest_upload(
                    project, filename=info.filename, data=zf.read(info),
                ))
    except zipfile.BadZipFile as e:
        return [FileResult(filename=filename, ok=False, error=f"bad zip: {e}")]
    return results