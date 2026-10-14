"""Seed a local development database.

Creates all tables (idempotent), then ensures a project with id=1 exists.
Safe to run repeatedly.
"""
from __future__ import annotations

from sqlalchemy import select

from app.db import SessionLocal, init_db
from app.models.project import (
    DEFAULT_DOC_TYPES,
    DEFAULT_FIELD_TYPES,
    Project,
)


def seed() -> None:
    init_db()

    db = SessionLocal()
    try:
        # Project id=1 — the one the frontend defaults to.
        p1 = db.get(Project, 1)
        if p1 is None:
            p1 = Project(
                id=1,
                name="local",
                doc_types=list(DEFAULT_DOC_TYPES),
                field_types=list(DEFAULT_FIELD_TYPES),
                label_map={},
                model_settings={"family": "mock", "device": "cpu"},
                max_long_edge=None,
                default_dpi=300,
                default_language="ne",
                max_upload_mb=100,
                require_doc_type_on_submit=True,
            )
            db.add(p1)
            db.flush()
            print(f"created project id={p1.id} name={p1.name!r}")
        else:
            print(f"project id=1 already exists: name={p1.name!r}")

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    seed()