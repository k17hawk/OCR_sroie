"""Import a JSONL file. ``overwrite=True`` replaces existing documents."""
from __future__ import annotations

import hashlib
import json
from typing import Any, BinaryIO

from pydantic import ValidationError

from app.core.finalize import finalize
from app.schemas.doc_v2 import Doc
from app.services.projects import Project
from app.storage import doc_store, get_storage


def image_sha256(image_key: str) -> str:
    """Hash of the stored page image, so uploading the same image later
    finds this document instead of creating a blank copy. Empty if the
    image isn't in storage."""
    storage = get_storage()
    try:
        if not image_key or not storage.exists(image_key):
            return ""
        return hashlib.sha256(storage.get(image_key)).hexdigest()
    except Exception:
        return ""


def import_jsonl(
    project: Project,
    stream: BinaryIO,
    *,
    allow_generated: bool = True,
    overwrite: bool = False,
) -> dict[str, int]:
    imported = skipped = 0
    errors: list[str] = []

    for lineno, raw in enumerate(stream, start=1):
        if not raw.strip():
            continue
        try:
            payload: dict[str, Any] = json.loads(raw)
            doc, _ = doc_store.doc_from_record(payload)
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(f"line {lineno}: {e}")
            skipped += 1
            continue

        if doc.annotation.status == "generated" and not allow_generated:
            skipped += 1
            continue

        existing = doc_store.load(doc.document_id)
        if existing is not None and not overwrite:
            skipped += 1
            continue

        finalized = finalize(doc)

        if existing is not None:
            existing.doc = finalized
            existing.status = finalized.annotation.status
            existing.revision += 1
            doc_store.save(existing)
        else:
            n = doc_store.next_document_number()
            row = doc_store.DocumentRow(
                document_id=finalized.document_id,
                project_id=project.id,
                doc_number=n,
                revision=1,
                status=finalized.annotation.status,
                doc=finalized,
                image_key=finalized.image,
                sha256=image_sha256(finalized.image),
                source_filename="<imported>",
            )
            doc_store.save(row)
        imported += 1

    return {"imported": imported, "skipped": skipped, "errors": len(errors)}