"""Monotonic document-number allocation, safe across concurrent writers."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document_counter import DocumentCounter


def next_document_number(db: Session) -> int:
    row = db.execute(
        select(DocumentCounter).where(DocumentCounter.id == 1).with_for_update()
    ).scalar_one_or_none()
    if row is None:
        row = DocumentCounter(id=1, next_number=1)
        db.add(row)
        db.flush()
    n = row.next_number
    row.next_number = n + 1
    db.flush()
    return n


def format_document_id(n: int) -> str:
    return f"doc_{n:07d}"


def format_image_path(n: int, ext: str = "jpg") -> str:
    return f"images/page_{n:07d}.{ext}"


def format_original_path(n: int, source_ext: str) -> str:
    return f"originals/page_{n:07d}{source_ext}"
