"""Streaming exporters."""
from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import PurePosixPath

from app.core.finalize import finalize
from app.storage import doc_store
from app.storage import get_storage


def _zip_image_name(image: str) -> str:
    """Path of a page image inside the export zip. Uses the page's own image
    file name, which is unique per page (document_id is shared by all pages
    of a multi-page document)."""
    return f"images/{PurePosixPath(image).name}"


def stream_jsonl(
    project_id: int, *, statuses: list[str] | None = None,
    zip_image_paths: bool = False,
) -> Iterator[bytes]:
    for row in doc_store.list_for_project(project_id):
        if statuses and row.status not in statuses:
            continue
        doc = finalize(row.doc)
        if zip_image_paths:
            # training reads <folder of the jsonl>/<image>, so point at the
            # file as it is stored in the zip
            doc = doc.model_copy(update={"image": _zip_image_name(doc.image)})
        yield (doc.model_dump_json() + "\n").encode("utf-8")


def stream_zip_with_images(
    project_id: int, *, statuses: list[str] | None = None
) -> Iterator[bytes]:
    storage = get_storage()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        jsonl = b"".join(stream_jsonl(project_id, statuses=statuses, zip_image_paths=True))
        zf.writestr("labels.jsonl", jsonl)
        for row in doc_store.list_for_project(project_id):
            if statuses and row.status not in statuses:
                continue
            try:
                zf.writestr(
                    _zip_image_name(row.doc.image), storage.get(row.image_key)
                )
            except FileNotFoundError:
                continue
    yield buf.getvalue()