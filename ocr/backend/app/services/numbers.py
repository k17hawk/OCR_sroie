"""Monotonic document-number allocation, safe across concurrent writers."""
from __future__ import annotations

import re
from pathlib import PurePath

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


_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def safe_stem(filename: str) -> str:
    """File name without folders or extension, safe as a folder name.
    'scans/pdf_001.pdf' -> 'pdf_001'. Devanagari and other letters are kept."""
    stem = PurePath(filename.replace("\\", "/")).stem
    stem = _UNSAFE.sub("_", stem).strip(" .")
    return stem or "upload"


def page_base_key(stem: str, page_number: int) -> str:
    """Storage key (no extension) shared by a page's image and its .jsonl:
    manual_annotation/pdf_001/pdf_001_page1"""
    from app.config import get_settings
    folder = get_settings().annotation_dir.strip("/")
    return f"{folder}/{stem}/{stem}_page{page_number}"


def format_original_path(n: int, source_ext: str) -> str:
    return f"originals/page_{n:07d}{source_ext}"
