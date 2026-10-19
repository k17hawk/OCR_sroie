"""File-backed document storage.

Layout:
    <storage_root>/manual_annotation/pdf_001/pdf_001_page1.jsonl
                                              -- the annotated Doc of an uploaded
                                                 page (one JSON line); its image
                                                 pdf_001_page1.png sits next to it
    data/_db/documents/doc_0000001.json       -- the annotated Doc (documents
                                                 created before per-file folders)
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

import logging

from app.config import get_settings

log = logging.getLogger(__name__)
from app.core.migrate import migrate
from app.schemas.doc_v2 import Doc
from app.services.geometry import ensure_quads


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
    # storage key of the page's .jsonl (manual_annotation/<name>/<name>_pageN.jsonl);
    # None for documents stored the old way in _db/documents/
    doc_key: str | None = None
    source_stem: str | None = None
    # what migrate() changed when this page was loaded (not stored)
    load_warnings: list[dict[str, Any]] = field(default_factory=list)
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


def key_path(key: str) -> Path:
    """Absolute path of a storage key under storage_root."""
    root = Path(get_settings().storage_root).resolve()
    p = (root / key).resolve()
    if root not in p.parents:
        raise ValueError(f"path outside storage root: {key!r}")
    return p


def _row_doc_path(document_id: str, doc_key: str | None) -> Path:
    return key_path(doc_key) if doc_key else _doc_path(document_id)


def doc_from_record(data: dict[str, Any]) -> tuple[Doc, list[dict[str, Any]]]:
    """Old page record -> current Doc, plus what migrate() changed."""
    data, warnings = migrate(data)
    # ensure_quads: old files may have missing or stale quads.
    return Doc.model_validate(ensure_quads(data)), warnings


def read_doc_file_with_warnings(path: Path) -> tuple[Doc, list[dict[str, Any]]]:
    """A page .jsonl holds one JSON line; a pretty-printed (hand-edited)
    file is accepted too."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        data = json.loads(first)
    return doc_from_record(data)


def read_doc_file(path: Path) -> Doc:
    return read_doc_file_with_warnings(path)[0]


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

    doc_json = row.doc.model_dump(mode="json")
    if row.doc_key:
        # one line, so the page files can be concatenated into labels.jsonl
        _atomic_write(key_path(row.doc_key),
                      json.dumps(doc_json, ensure_ascii=False) + "\n")
    else:
        _atomic_write(_doc_path(row.document_id),
                      json.dumps(doc_json, indent=2, ensure_ascii=False) + "\n")

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
        "doc_key": row.doc_key,
        "source_stem": row.source_stem,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
    _atomic_write(
        _meta_path(row.document_id),
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
    )


def load(document_id: str) -> DocumentRow | None:
    meta_file = _meta_path(document_id)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    doc_file = _row_doc_path(document_id, meta.get("doc_key"))
    if not doc_file.exists():
        return None

    doc, load_warnings = read_doc_file_with_warnings(doc_file)
    if load_warnings:
        log.info("%s: migrated on load:\n  %s", document_id,
                 "\n  ".join(w["message"] for w in load_warnings))

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
        doc_key=meta.get("doc_key"),
        source_stem=meta.get("source_stem"),
        load_warnings=load_warnings,
        created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
        updated_at=meta.get("updated_at", datetime.now(timezone.utc).isoformat()),
    )

    if external_edit:
        save(row)

    return row

def delete(document_id: str) -> None:
    """Removes the document from the tool. A page stored in manual_annotation/
    keeps its .jsonl and image on disk; uploading the file again brings it back."""
    for p in (_doc_path(document_id), _meta_path(document_id)):
        if p.exists():
            p.unlink()


def rows_for_stem(project_id: int | None, stem: str) -> list[DocumentRow]:
    """Pages created from the upload named <stem>, by page number.
    project_id None searches every project."""
    root = _root() / "documents"
    out: list[DocumentRow] = []
    for meta_path in root.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if meta.get("source_stem") != stem:
            continue
        if project_id is not None and int(meta.get("project_id", -1)) != project_id:
            continue
        try:
            row = load(meta_path.name[: -len(".meta.json")])
        except ValueError as e:
            log.error("skipping %s: %s", meta_path.name, e)
            continue
        if row is not None:
            out.append(row)
    out.sort(key=lambda r: r.page_number)
    return out


def ids_for_doc_key(doc_key: str) -> list[str]:
    """Every document whose metadata points at this page file (normally
    one; more after a page's .jsonl was deleted by hand and uploaded again)."""
    root = _root() / "documents"
    out: list[str] = []
    for meta_path in sorted(root.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if meta.get("doc_key") == doc_key:
            out.append(meta_path.name[: -len(".meta.json")])
    return out


def _retire(document_id: str, why: str) -> None:
    """Take a duplicate entry out of the list. Its metadata is kept next to
    the others as <id>.meta.json.orphan, so nothing is lost; the page file
    itself belongs to the entry that was kept."""
    meta = _meta_path(document_id)
    if meta.exists():
        os.replace(meta, meta.with_name(meta.name + ".orphan"))
    log.warning("%s: %s; moved its metadata to %s.meta.json.orphan", document_id, why, document_id)
    append_audit({"event": "retire_duplicate", "document_id": document_id, "reason": why})


def _dedupe_by_doc_key(rows: list[DocumentRow]) -> list[DocumentRow]:
    """Entries that share one page file are one page. Keep the entry the
    file itself names (its ``document_id``), else the most recently saved."""
    by_key: dict[str, list[DocumentRow]] = {}
    for r in rows:
        if r.doc_key:
            by_key.setdefault(r.doc_key, []).append(r)
    drop: set[str] = set()
    for key, group in by_key.items():
        if len(group) < 2:
            continue
        named = [r for r in group if r.document_id == r.doc.document_id]
        keep = named[0] if named else max(group, key=lambda r: r.updated_at)
        for r in group:
            if r is not keep:
                _retire(r.document_id, f"same page file {key} as {keep.document_id}")
                drop.add(r.document_id)
    return [r for r in rows if r.document_id not in drop]


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
        try:
            row = load(did)
        except ValueError as e:      # a hand-edited .jsonl that no longer parses
            log.error("skipping %s: %s", did, e)
            continue
        if row is not None:
            out.append(row)
    out = _dedupe_by_doc_key(out)
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