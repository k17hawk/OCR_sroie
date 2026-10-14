"""Document accessors on top of doc_store."""
from __future__ import annotations

from app.storage import doc_store


def get(document_id: str) -> doc_store.DocumentRow | None:
    return doc_store.load(document_id)


def list_for_project(project_id: int) -> list[doc_store.DocumentRow]:
    return doc_store.list_for_project(project_id)


def save(row: doc_store.DocumentRow) -> None:
    doc_store.save(row)


def delete(document_id: str) -> None:
    doc_store.delete(document_id)


def next_document_number() -> int:
    return doc_store.next_document_number()


def find_by_sha256(project_id: int, sha256: str) -> doc_store.DocumentRow | None:
    return doc_store.find_by_sha256(project_id, sha256)