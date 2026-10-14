"""Bring the local SQLite schema up to date with the ORM models.

For each table declared in ``Base.metadata``:
  * If the table doesn't exist, create it.
  * If it exists but is missing columns, ALTER TABLE ADD COLUMN for each.
  * Finally, backfill Python-side JSON defaults on any rows where they're NULL.

Idempotent. Never drops or renames anything.

For local dev only. In production, use Alembic.
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect

from app.config import get_settings
from app.db import Base, _engine
import app.models  # noqa: F401  — ensures every model is registered


def _sqlite_path() -> Path:
    url = get_settings().db_url
    if not url.startswith("sqlite"):
        raise SystemExit(f"this script only handles sqlite:// URLs, got {url!r}")
    # Handles:  sqlite:///./foo.db   and   sqlite:////abs/path.db
    _, _, path = url.partition(":///")
    return Path(path).resolve()


def sync_columns() -> list[tuple[str, str]]:
    """Add any missing columns. Returns list of (table, column) added."""
    engine = _engine
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    added: list[tuple[str, str]] = []

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                table.create(conn)
                added.append((table.name, "<whole table>"))
                continue

            db_cols = {c["name"] for c in insp.get_columns(table.name)}
            for column in table.columns:
                if column.name in db_cols:
                    continue
                ddl_type = column.type.compile(dialect=engine.dialect)
                if not column.nullable and column.default is None:
                    raise SystemExit(
                        f"cannot add NOT NULL column {table.name}.{column.name} "
                        f"without a default — write an Alembic migration instead"
                    )
                conn.exec_driver_sql(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'
                )
                added.append((table.name, column.name))

    return added


def backfill_defaults() -> None:
    """Fill NULL JSON columns with the Python-side defaults on the model."""
    from app.db import SessionLocal
    from app.models.project import (
        DEFAULT_DOC_TYPES,
        DEFAULT_FIELD_TYPES,
        DEFAULT_LAYOUT_TYPES,
        Project,
    )

    db = SessionLocal()
    try:
        for p in db.query(Project).all():
            changed = False
            if not p.doc_types:
                p.doc_types = list(DEFAULT_DOC_TYPES)
                changed = True
            if not p.field_types:
                p.field_types = list(DEFAULT_FIELD_TYPES)
                changed = True
            if not p.layout_types:
                p.layout_types = list(DEFAULT_LAYOUT_TYPES)
                changed = True
            if not p.model_settings:
                p.model_settings = {"family": "mock", "device": "cpu"}
                changed = True
            if changed:
                db.add(p)
                print(f"  backfilled project id={p.id}")
        db.commit()
    finally:
        db.close()


def main() -> None:
    print(f"database: {_sqlite_path()}")
    added = sync_columns()
    if added:
        print("added:")
        for t, c in added:
            print(f"  {t}.{c}")
    else:
        print("schema already in sync")
    print("backfilling defaults…")
    backfill_defaults()
    print("done.")


if __name__ == "__main__":
    main()