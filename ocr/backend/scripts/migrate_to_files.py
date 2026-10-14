"""Dump every row from the legacy SQLite DB to file storage.

Safe to run once. On subsequent runs it does nothing (files already exist).
Run from ``backend/``:  ``python -m scripts.migrate_to_files``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from app.config import get_settings
from app.schemas.doc_v2 import Doc
from app.services.projects import Project, _write_all
from app.storage import doc_store


def _coerce_json(value: Any, default: Any) -> Any:
    """Return a real Python object.

    SQLAlchemy sometimes hands back JSON columns as strings (SQLite's
    text affinity). Accept either: a decoded object, a JSON string, or
    None → default.
    """
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def main() -> None:
    settings = get_settings()
    db_path = Path(settings.db_url.replace("sqlite:///", "")).resolve()
    if not db_path.exists():
        print(f"no legacy DB at {db_path} — nothing to do")
        return

    engine = create_engine(settings.db_url)
    with engine.connect() as conn:
        # --- projects ---
        projects: list[Project] = []
        rows = conn.execute(text(
            "SELECT id, name, doc_types, field_types, layout_types, label_map, "
            "model_config, max_long_edge, default_dpi, default_language, "
            "max_upload_mb, require_doc_type_on_submit, created_at, updated_at "
            "FROM projects"
        ))
        for r in rows:
            projects.append(Project(
                id=r.id,
                name=r.name,
                doc_types=_coerce_json(r.doc_types, []),
                field_types=_coerce_json(r.field_types, []),
                layout_types=_coerce_json(r.layout_types, []),
                label_map=_coerce_json(r.label_map, {}),
                model_settings=_coerce_json(r.model_config, {}),
                max_long_edge=r.max_long_edge,
                default_dpi=r.default_dpi or 200,
                default_language=r.default_language or "ne",
                max_upload_mb=r.max_upload_mb or 100,
                require_doc_type_on_submit=bool(r.require_doc_type_on_submit),
                created_at=str(r.created_at),
                updated_at=str(r.updated_at),
            ))
        if projects:
            _write_all(projects)
        print(f"projects: {len(projects)}")

        # --- documents ---
        count = 0
        rows = conn.execute(text(
            "SELECT project_id, document_id, doc_number, revision, status, "
            "image_key, sha256, source_file_id, source_filename, text_layer, "
            "legacy_font, doc_json, created_at, updated_at FROM documents"
        ))
        for r in rows:
            try:
                doc = Doc.model_validate(_coerce_json(r.doc_json, {}))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {r.document_id}: {e}")
                continue

            row = doc_store.DocumentRow(
                document_id=r.document_id,
                project_id=r.project_id,
                doc_number=r.doc_number,
                revision=r.revision,
                status=r.status,
                doc=doc,
                image_key=r.image_key,
                sha256=r.sha256 or "",
                source_file_id=r.source_file_id,
                source_filename=r.source_filename,
                text_layer=r.text_layer,
                legacy_font=r.legacy_font,
                created_at=str(r.created_at),
                updated_at=str(r.updated_at),
            )
            doc_store.save(row)
            count += 1
        print(f"documents: {count}")

        # --- counter ---
        try:
            n = conn.execute(
                text("SELECT next_number FROM document_counter WHERE id=1")
            ).scalar()
        except Exception:
            n = None
        if n is not None:
            root = Path(settings.storage_root) / "_db"
            doc_store._atomic_write(
                root / "counter.json",
                json.dumps({"next_number": n}) + "\n",
            )

    print("done")


if __name__ == "__main__":
    main()