"""Workflow status transitions."""
from __future__ import annotations

from fastapi import HTTPException

from app.services.projects import get as get_project
from app.storage import doc_store


_ALLOWED: dict[str, set[str]] = {
    "generated":     {"pre_annotated"},
    "pre_annotated": {"in_progress"},
    "in_progress":   {"submitted"},
    "submitted":     {"approved", "rejected"},
    "approved":      {"submitted"},
    "rejected":      {"in_progress", "submitted"},
}


def change(
    *,
    row: doc_store.DocumentRow,
    target: str,
    user_id: int | None,
    comment: str | None = None,
) -> doc_store.DocumentRow:
    current = row.status
    if target not in _ALLOWED.get(current, set()):
        raise HTTPException(409, f"cannot transition {current!r} -> {target!r}")

    if target == "submitted":
        project = get_project(row.project_id)
        if project and project.require_doc_type_on_submit:
            if row.doc.doc_type in (None, "", "other"):
                raise HTTPException(409, "doc_type must be set before submit")

    row.status = target
    row.doc.annotation.status = target
    row.revision += 1

    doc_store.save(row)
    doc_store.append_audit({
        "user_id": user_id,
        "document_id": row.document_id,
        "action": target,
        "comment": comment or "",
    })
    return row