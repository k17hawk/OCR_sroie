"""Streaming exporters."""
from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator

from app.core.finalize import finalize
from app.storage import doc_store
from app.storage import get_storage


def stream_jsonl(
    project_id: int, *, statuses: list[str] | None = None
) -> Iterator[bytes]:
    for row in doc_store.list_for_project(project_id):
        if statuses and row.status not in statuses:
            continue
        doc = finalize(row.doc)
        yield (doc.model_dump_json() + "\n").encode("utf-8")


def stream_zip_with_images(
    project_id: int, *, statuses: list[str] | None = None
) -> Iterator[bytes]:
    storage = get_storage()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        jsonl = b"".join(stream_jsonl(project_id, statuses=statuses))
        zf.writestr("annotations.jsonl", jsonl)
        for row in doc_store.list_for_project(project_id):
            if statuses and row.status not in statuses:
                continue
            try:
                zf.writestr(
                    f"images/{row.document_id}.jpg", storage.get(row.image_key)
                )
            except FileNotFoundError:
                continue
    yield buf.getvalue()