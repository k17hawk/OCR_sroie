"""Import a JSONL file. ``overwrite=True`` replaces existing documents."""
from __future__ import annotations

import json
from typing import Any, BinaryIO

from pydantic import ValidationError

from app.core.finalize import finalize
from app.schemas.doc_v2 import Doc
from app.services.projects import Project
from app.storage import doc_store


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
            doc = Doc.model_validate(payload)
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
                source_filename="<imported>",
            )
            doc_store.save(row)
        imported += 1

    return {"imported": imported, "skipped": skipped, "errors": len(errors)}