"""File-backed document storage.

Layout:
    data/_db/documents/doc_0000001.json       -- the annotated Doc
    data/_db/documents/doc_0000001.meta.json  -- sidecar metadata
    data/_db/counter.json                     -- next_document_number
    data/_db/audit.jsonl                      -- append-only log

The Doc file is byte-identical to what ``GET /documents/{id}`` returns
under the ``doc`` key, and to one line of a JSONL export. Edit it by hand
and the next request sees your change.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.schemas.doc_v2 import Doc


# ---------------------------------------------------------------------------
# Row
# ---------------------------------------------------------------------------

@dataclass
class DocumentRow:
    document_id: str
    project_id: int
    doc_number: int
    revision: int
    status: str
    doc: Doc
    image_key: str
    sha256: str = ""
    source_file_id: str | None = None
    source_filename: str | None = None
    text_layer: str | None = None
    legacy_font: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- compatibility with the SQLAlchemy-era call sites ---------------

    @property
    def id(self) -> int:
        return self.doc_number

    @property
    def page_number(self) -> int:
        return self.doc.page_number

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    @property
    def width(self) -> int:
        return self.doc.width

    @property
    def height(self) -> int:
        return self.doc.height

    @property
    def doc_json(self) -> dict[str, Any]:
        return self.doc.model_dump(mode="json")

    @doc_json.setter
    def doc_json(self, value: dict[str, Any]) -> None:
        self.doc = Doc.model_validate(value)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _root() -> Path:
    root = Path(get_settings().storage_root) / "_db"
    (root / "documents").mkdir(parents=True, exist_ok=True)
    return root


def _doc_path(document_id: str) -> Path:
    return _root() / "documents" / f"{document_id}.json"


def _meta_path(document_id: str) -> Path:
    return _root() / "documents" / f"{document_id}.meta.json"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save(row: DocumentRow) -> None:
    row.updated_at = datetime.now(timezone.utc).isoformat()

    _atomic_write(
        _doc_path(row.document_id),
        json.dumps(row.doc.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
    )

    meta = {
        "document_id": row.document_id,
        "project_id": row.project_id,
        "doc_number": row.doc_number,
        "revision": row.revision,
        "status": row.status,
        "image_key": row.image_key,
        "sha256": row.sha256,
        "source_file_id": row.source_file_id,
        "source_filename": row.source_filename,
        "text_layer": row.text_layer,
        "legacy_font": row.legacy_font,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
    _atomic_write(
        _meta_path(row.document_id),
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
    )


def load(document_id: str) -> DocumentRow | None:
    doc_file = _doc_path(document_id)
    if not doc_file.exists():
        return None

    doc = Doc.model_validate(json.loads(doc_file.read_text(encoding="utf-8")))

    meta_file = _meta_path(document_id)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    # Auto-detect a hand-edit of the Doc file and bump revision so
    # optimistic locking stays honest.
    doc_mtime = doc_file.stat().st_mtime
    meta_mtime = meta_file.stat().st_mtime if meta_file.exists() else 0.0
    external_edit = doc_mtime > meta_mtime + 0.5

    row = DocumentRow(
        document_id=document_id,
        project_id=int(meta.get("project_id", 1)),
        doc_number=int(meta.get("doc_number", _parse_number(document_id))),
        revision=int(meta.get("revision", 1)) + (1 if external_edit else 0),
        status=meta.get("status", doc.annotation.status),
        doc=doc,
        image_key=meta.get("image_key", doc.image),
        sha256=meta.get("sha256", ""),
        source_file_id=meta.get("source_file_id"),
        source_filename=meta.get("source_filename"),
        text_layer=meta.get("text_layer"),
        legacy_font=meta.get("legacy_font"),
        created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
        updated_at=meta.get("updated_at", datetime.now(timezone.utc).isoformat()),
    )

    if external_edit:
        save(row)

    return row


def delete(document_id: str) -> None:
    for p in (_doc_path(document_id), _meta_path(document_id)):
        if p.exists():
            p.unlink()


def list_for_project(project_id: int) -> list[DocumentRow]:
    root = _root() / "documents"
    out: list[DocumentRow] = []
    for meta_path in sorted(root.glob("*.meta.json")):
        did = meta_path.name[: -len(".meta.json")]
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if int(meta.get("project_id", -1)) != project_id:
            continue
        row = load(did)
        if row is not None:
            out.append(row)
    out.sort(key=lambda r: r.doc_number)
    return out


def _fill_missing_sha256(row: DocumentRow) -> None:
    """Documents imported from JSONL were saved without a hash, so an
    upload of the same image could never match them. Hash the stored image
    the first time it's needed and remember it."""
    if row.sha256 or not row.image_key:
        return
    from app.storage import get_storage  # local import: avoid a cycle

    storage = get_storage()
    try:
        if not storage.exists(row.image_key):
            return
        row.sha256 = hashlib.sha256(storage.get(row.image_key)).hexdigest()
    except Exception:
        return
    save(row)


def index_by_sha256(project_id: int) -> dict[str, DocumentRow]:
    """hash -> document for every document in the project."""
    out: dict[str, DocumentRow] = {}
    for row in list_for_project(project_id):
        _fill_missing_sha256(row)
        if row.sha256:
            out.setdefault(row.sha256, row)
    return out


def find_by_sha256(project_id: int, sha256: str) -> DocumentRow | None:
    return index_by_sha256(project_id).get(sha256)


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

def next_document_number() -> int:
    path = _root() / "counter.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"next_number": 1}
    n = int(state["next_number"])
    state["next_number"] = n + 1
    _atomic_write(path, json.dumps(state) + "\n")
    return n


def _parse_number(document_id: str) -> int:
    try:
        return int(document_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return 1


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def append_audit(entry: dict[str, Any]) -> None:
    path = _root() / "audit.jsonl"
    entry = {**entry, "at": datetime.now(timezone.utc).isoformat()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")