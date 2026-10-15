"""Upload orchestration: validate, dedupe, store, create skeleton documents."""
from __future__ import annotations

import hashlib
import io
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.models.project import DEFAULT_LAYOUT_TYPES  # not used; kept for reference
from app.services import images as img_service
from app.services import pdfs as pdf_service
from app.services import quality_metrics
from app.services.numbers import page_base_key, safe_stem
from app.services.projects import Project
from app.services.skeleton import make_skeleton
from app.services.numbers import format_document_id
from app.storage import doc_store, get_storage


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
    warning: str | None = None


def ingest_upload(project: Project, *, filename: str, data: bytes) -> list[FileResult]:
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_ZIP_EXTS:
        return _ingest_zip(project, filename, data)
    if ext in ALLOWED_PDF_EXTS:
        return [_ingest_pdf(project, filename, data)]
    return [_ingest_image(project, filename, data)]


# ---------------------------------------------------------------------------
# Per-file folders
#
# Every upload lives in <storage_root>/manual_annotation/<name>/ :
#     <name>_page1.png    the page image all coordinates refer to
#     <name>_page1.jsonl  the page's annotation (one JSON line), loaded and
#                         saved by the editor
# Uploading a file with the same name again opens those pages. A page whose
# .jsonl is already on disk (copied in, or kept after the tool's index was
# reset) is loaded from it instead of starting empty.
# ---------------------------------------------------------------------------

_IMAGE_EXTS_ON_DISK = ("png", "jpg", "jpeg")


def _existing_image_key(base: str) -> str | None:
    storage = get_storage()
    for ext in _IMAGE_EXTS_ON_DISK:
        if storage.exists(f"{base}.{ext}"):
            return f"{base}.{ext}"
    return None


def _free_document_id(document_id: str) -> bool:
    return (not doc_store._meta_path(document_id).exists()
            and not doc_store._doc_path(document_id).exists())


def _page_row(
    project: Project, *, stem: str, page_number: int, page_count: int,
    get_page, sha256: str, filename: str, source_file_id: str | None,
) -> tuple[doc_store.DocumentRow, str | None]:
    """Create or load one page. ``get_page()`` returns
    (PreparedImage, original_bytes, text_layer, legacy_font) and is only
    called when the image has to be written. Returns (row, warning)."""
    base = page_base_key(stem, page_number)
    doc_key = f"{base}.jsonl"
    doc_file = doc_store.key_path(doc_key)
    image_key = _existing_image_key(base)
    text_layer = legacy = None
    warning = None

    if doc_file.exists():
        doc = doc_store.read_doc_file(doc_file)     # ValueError on a bad file
        if image_key is None:
            prepared, _, text_layer, legacy = get_page()
            image_key = img_service.store_image_at(base, prepared)
            if (prepared.width, prepared.height) != (doc.width, doc.height):
                warning = (f"page {page_number}: re-rendered image is {prepared.width}x"
                           f"{prepared.height} but {doc_key} expects {doc.width}x{doc.height}")
        if _free_document_id(doc.document_id):
            document_id = doc.document_id
            n = doc_store._parse_number(document_id)
        else:
            n = doc_store.next_document_number()
            document_id = format_document_id(n)
        doc = doc.model_copy(update={
            "image": image_key, "document_id": document_id,
            "page_number": page_number, "page_count": page_count,
        })
    else:
        prepared, orig, text_layer, legacy = get_page()
        if image_key is None:
            image_key = img_service.store_image_at(base, prepared)
        n = doc_store.next_document_number()
        doc = make_skeleton(
            n=n, image_ext=prepared.save_ext,
            width=prepared.width, height=prepared.height,
            page_number=page_number, page_count=page_count,
            language=project.default_language,
            quality=quality_metrics.compute_quality(prepared.image, orig),
        ).model_copy(update={"image": image_key})
        document_id = doc.document_id

    row = doc_store.DocumentRow(
        document_id=document_id,
        project_id=project.id,
        doc_number=n,
        revision=1,
        status=doc.annotation.status,
        doc=doc,
        image_key=image_key,
        sha256=sha256,
        source_file_id=source_file_id,
        source_filename=filename,
        text_layer=text_layer or None,
        legacy_font=legacy,
        doc_key=doc_key,
        source_stem=stem,
    )
    doc_store.save(row)
    return row, warning


def _saved_upload(
    project: Project, filename: str, stem: str, page_sha: list[str],
) -> FileResult | None:
    """Result for a file name that was uploaded before, or None if it is new
    (or some pages are still missing and must be created)."""
    rows = doc_store.rows_for_stem(project.id, stem)
    if not rows:
        other = doc_store.rows_for_stem(None, stem)
        if other:
            return FileResult(
                filename=filename, ok=False,
                error=(f"'{stem}' is already used by project {other[0].project_id} "
                       f"in {page_base_key(stem, 1).rsplit('/', 1)[0]}/; rename the file"),
            )
        return None
    same = [r.sha256 for r in rows] == page_sha[: len(rows)]
    if same and len(rows) < len(page_sha):
        return None          # an earlier upload stopped part-way: add the rest
    result = FileResult(
        filename=filename, ok=True, skipped_duplicate=True,
        document_ids=[r.document_id for r in rows],
    )
    if not same:
        result.warning = (f"{filename}: a file with this name was uploaded before with "
                          f"different content; opened its saved annotations "
                          f"(the new file was not stored)")
    return result


def _pages_result(filename, rows, warnings) -> FileResult:
    return FileResult(
        filename=filename, ok=True,
        document_ids=[r.document_id for r in rows],
        warning="; ".join(warnings) or None,
    )


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

    stem = safe_stem(filename)
    saved = _saved_upload(project, filename, stem, [prepared.sha256])
    if saved is not None:
        return saved

    # the same image uploaded under another name before per-file folders existed
    existing = doc_store.find_by_sha256(project.id, prepared.sha256)
    if existing is not None and not existing.source_stem:
        return FileResult(
            filename=filename, ok=True, skipped_duplicate=True,
            document_ids=[existing.document_id],
        )

    try:
        row, warning = _page_row(
            project, stem=stem, page_number=1, page_count=1,
            get_page=lambda: (prepared, data, None, None),
            sha256=prepared.sha256, filename=filename, source_file_id=None,
        )
    except ValueError as e:
        return FileResult(filename=filename, ok=False, error=f"{stem}_page1.jsonl: {e}")
    return _pages_result(filename, [row], [warning] if warning else [])


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def _ingest_pdf(project: Project, filename: str, data: bytes) -> FileResult:
    if len(data) > project.max_upload_mb * 1024 * 1024:
        return FileResult(filename=filename, ok=False,
                          error=f"file exceeds {project.max_upload_mb} MB")

    try:
        page_count = pdf_service.pdf_page_count(data)
    except Exception as e:
        return FileResult(filename=filename, ok=False, error=f"pdf: {e}")

    pdf_hash = hashlib.sha256(data).hexdigest()
    page_sha = [f"pdf:{pdf_hash}:p{i}" for i in range(1, page_count + 1)]
    stem = safe_stem(filename)

    saved = _saved_upload(project, filename, stem, page_sha)
    if saved is not None:
        return saved

    # The same PDF uploaded before per-file folders existed: open those pages.
    index = doc_store.index_by_sha256(project.id)
    old = [index.get(h) for h in page_sha]
    if page_count and all(old) and not any(r.source_stem for r in old):
        return FileResult(
            filename=filename, ok=True, skipped_duplicate=True,
            document_ids=[r.document_id for r in old],
        )

    done = {r.page_number: r for r in doc_store.rows_for_stem(project.id, stem)}
    todo = [p for p in range(1, page_count + 1) if p not in done]
    need_render = [p for p in todo
                   if _existing_image_key(page_base_key(stem, p)) is None]

    rendered: dict[int, tuple] = {}
    if need_render:
        try:
            for page_number, png_bytes, text_layer, legacy in pdf_service.iter_pdf_pages(
                data, dpi=project.default_dpi, image_format="PNG",
            ):
                if page_number not in need_render:
                    continue
                prepared = img_service.prepare_image(
                    data=png_bytes, source_ext=".png",
                    max_long_edge=project.max_long_edge,
                )
                rendered[page_number] = (prepared, png_bytes, text_layer, legacy)
        except img_service.ImageError as e:
            return FileResult(filename=filename, ok=False, error=f"page: {e}")
        except Exception as e:
            return FileResult(filename=filename, ok=False, error=f"pdf: {e}")

    source_file_id = (next(iter(done.values())).source_file_id if done
                      else f"src_{uuid.uuid4().hex[:12]}")
    warnings: list[str] = []
    for page_number in todo:
        try:
            row, warning = _page_row(
                project, stem=stem, page_number=page_number, page_count=page_count,
                get_page=lambda p=page_number: rendered[p],
                sha256=page_sha[page_number - 1], filename=filename,
                source_file_id=source_file_id,
            )
        except ValueError as e:
            return FileResult(filename=filename, ok=False,
                              error=f"{stem}_page{page_number}.jsonl: {e}")
        done[page_number] = row
        if warning:
            warnings.append(warning)
        rendered.pop(page_number, None)

    return _pages_result(filename, [done[p] for p in sorted(done)], warnings)


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